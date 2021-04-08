import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import click
import os
import random

import wandb
from timing import Timing
from tqdm import tqdm
import matplotlib.pyplot as plt

from reward_predictor import RewardPredictor
from trajectories import preprocess_dataset, find_activations

PAD_VAL = 10
ACTION_SPACE_SIZE = 3
OBSERVATION_SPACE_DIMS = (3, 24, 24)


def evaluate(model, criterion, data_loader, device, timing, ca_gt, verbose=False):
    attention_threshold, batch_size, seq_len = ca_gt['attention_threshold'], ca_gt['batch_size'], ca_gt['seq_len']

    losses = []
    accuracies = []
    ca_precisions = []
    ca_recalls = []
    rel_attention_vals = torch.zeros((batch_size, seq_len*2+1), device=device)  # N, S*2+1

    batch_count = 0

    for batch_idx, batch in enumerate(data_loader):
        observations, actions, returns, indices = batch
        batch_count += 1

        trigger_activations_batch = ca_gt['trigger_activations'][indices]
        trigger_timesteps_batch = ca_gt['trigger_timesteps'][indices]
        prize_timesteps_batch = ca_gt['prize_timesteps'][indices]
        prize_episodes_mask_batch = ca_gt['episodes_with_prize_mask'][indices]
        trigger_episodes_mask_batch = ca_gt['episodes_with_trigger_mask'][indices]

        N, seq_len = observations.shape[:2]

        with Timing(timing, 'time_eval_preprocess'):
            observations = observations.transpose(
                2, 4).transpose(0, 1).to(device)
            actions = actions.transpose(0, 1).to(device)
            returns = returns.transpose(0, 1).to(device)

        with Timing(timing, 'time_eval_run_inference'):
            output, attention_output = model(observations, actions, output_attention=True)

        with Timing(timing, 'time_eval_calc_metrics'):
            # reshape for CrossEntropyLoss
            output = output.permute(0, 2, 1)
            returns = returns.transpose(0, 1)

            loss = criterion(output, returns)
            preds = output.argmax(dim=1)

            # don't take padded timesteps into account
            padding_mask = actions.transpose(0, 1)[:, :, 0] != PAD_VAL
            masked_preds = preds[padding_mask]
            masked_returns = returns[padding_mask]

            acc = torch.sum(masked_preds == masked_returns) / \
                masked_returns.numel()

            # Compute credit assignment average graph #

            # for some weird reason, attention_matrices[:, prize_timesteps_batch] results in [N, N, S] rather than [N, S]
            # fixed with help of:
            # https://discuss.pytorch.org/t/selecting-element-on-dimension-from-list-of-indexes/36319/2?u=nick-baliesnyi
            attention_vals = attention_output[torch.arange(N), prize_timesteps_batch]

            # zero out episodes where prize or trigger wasn't touched
            attention_vals *= prize_episodes_mask_batch[:, None]

            # compute relative timesteps for each episode relative to trigger activation
            # e.g.
            # [[-10, -9, -8, ..., 39],
            # [-25, -24, -24,..., 24],
            #                     ...]
            rel_timesteps = torch.arange(0, seq_len) + seq_len
            rel_timesteps = rel_timesteps.repeat(
                N, 1).to(device)
            rel_timesteps -= trigger_timesteps_batch[:, None]

            # scatter the batch's attention values over a relative timestep matrix (N, S*2+1)
            # normalize it *now* because later we won't know the number of relevant episodes
            n_relevant_episodes = torch.sum(torch.logical_and(prize_episodes_mask_batch, trigger_episodes_mask_batch))
            rel_attention_vals_batch = torch.zeros_like(rel_attention_vals).scatter(1, rel_timesteps, attention_vals) / n_relevant_episodes

            # Compute credit assignment precision/recall #

            attention_discrete = torch.gt(attention_vals, attention_threshold, out=torch.empty(
                attention_vals.shape, dtype=torch.uint8, device=device))
            true_positives = torch.sum(torch.logical_and(attention_discrete, trigger_activations_batch))

            ca_precision = true_positives / torch.sum(attention_discrete)
            ca_recall = true_positives / n_relevant_episodes

        losses.append(loss.item())
        accuracies.append(acc.item())
        ca_precisions.append(ca_precision.item())
        ca_recalls.append(ca_recall.item())
        rel_attention_vals += rel_attention_vals_batch

        del observations
        del actions
        del returns
        del output

    mean_loss = sum(losses) / len(losses)
    mean_acc = sum(accuracies) / len(accuracies)
    mean_ca_precision = sum(ca_precisions) / len(ca_precisions)
    mean_ca_recall = sum(ca_recalls) / len(ca_recalls)
    mean_rel_attention_vals = torch.sum(rel_attention_vals, axis=0) / batch_count

    del losses
    del accuracies
    del ca_precisions
    del ca_recalls

    if verbose:
        print('mean_loss:', mean_loss)
        print('mean_acc:', mean_acc)

    return mean_loss, mean_acc, mean_ca_precision, mean_ca_recall, mean_rel_attention_vals


@click.command()
@click.option('--note', type=str, help='message to explain how is this run different', required=True)
@click.option('--data-path', type=click.Path(exists=True), help='path to trajectories dataset', required=True)
@click.option('--seed', type=int, default=42, help='random seed used')
@click.option('--seeds', type=int, default=1, help='# of random seeds used')
@click.option('--log-frequency', type=int, default=5e1, help='logging frequency, iterations')
@click.option('--learning-rate', type=float, default=3e-3, help='goal learning rate')
@click.option('--epochs', type=int, default=10, help='number of epochs to train for')
@click.option('--batch-size', default=128, help='training batch size')
@click.option('--attention-threshold', default=0.2, help='threshold attention weight discretization')
@click.option('--valid-size', type=float, default=0.2, help='proportion of validation set')
@click.option('--use-wandb/--no-wandb', default=True)
def train(note, data_path, seed, seeds, log_frequency,
          learning_rate, epochs, batch_size,
          attention_threshold, valid_size, use_wandb):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    wandb.login()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data
    dataset = torch.load(data_path)

    trigger_activations = find_activations(dataset.observations, dataset.actions, target='trigger').to(device)  # (N, S)
    prize_activations = find_activations(dataset.observations, dataset.actions, target='prize')  # (N, S)

    trigger_timesteps = torch.argmax(trigger_activations, axis=-1).to(device)  # (N,)
    prize_timesteps = torch.argmax(prize_activations, axis=-1)  # (N,)

    # episode binary masks where prizes/triggers were taken
    episodes_with_trigger_mask = torch.sum(trigger_activations, axis=-1).to(device)  # (N, )
    episodes_with_prize_mask = torch.sum(prize_activations, axis=-1).to(device)  # (N, )

    seq_len = dataset.observations.shape[1]
    rel_attention_vals = torch.zeros((batch_size, seq_len*2+1), device=device)  # N, S*2+1

    # credit assignment ground truths
    ca_gt = {
        'attention_threshold': attention_threshold,
        'batch_size': batch_size,
        'episodes_with_trigger_mask': episodes_with_trigger_mask,
        'episodes_with_prize_mask': episodes_with_prize_mask,
        'seq_len': seq_len,
        'trigger_activations': trigger_activations,
        'trigger_timesteps': trigger_timesteps,
        'prize_activations': prize_activations,
        'prize_timesteps': prize_timesteps,
    }

    train_loader, valid_loader = preprocess_dataset(dataset, data_path,
                                                    batch_size=batch_size,
                                                    valid_size=valid_size,
                                                    seed=seed)

    # model
    class_weights = torch.tensor([0.499, 0.02, 0.499]).to(device)

    for i_seed in range(seed, seed+seeds):
        print(f"> Experiment Seed {i_seed}")

        wandb.init(project='attentional_fw_baselines',
                   entity='ut-rl-credit',
                   tags=['prod'],
                   notes=note,
                   mode='online' if use_wandb else 'disabled',
                   config=dict(
                       attention_threshold=attention_threshold,
                       batch_size=batch_size,
                       data_path=data_path,
                       epochs=epochs,
                       learning_rate=learning_rate,
                       seed=i_seed,
                       valid_size=valid_size,
                   ))
        # Upload models at the end of training
        save_dir = wandb.run.dir if use_wandb else './'
        wandb.save(os.path.join(save_dir, "*.pt"))

        torch.cuda.manual_seed(i_seed)
        torch.manual_seed(i_seed)
        np.random.seed(i_seed)
        random.seed(i_seed)

        model = RewardPredictor(
            OBSERVATION_SPACE_DIMS, ACTION_SPACE_SIZE, device, verbose=False).to(device)

        wandb.watch(model)

        best_val_precision = 0
        best_val_recall = 0
        best_val_acc = 0
        best_val_loss = 10000

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=10, verbose=True
        )

        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=PAD_VAL)

        for epoch in range(epochs):
            print(f"> Epoch {epoch+1}/{epochs}", end=' ')

            losses = []

            pbar = tqdm(train_loader)

            for batch_idx, batch in enumerate(pbar):
                observations, actions, returns, indices = batch

                trigger_activations_batch = trigger_activations[indices]
                trigger_timesteps_batch = trigger_timesteps[indices]
                prize_timesteps_batch = prize_timesteps[indices]
                prize_episodes_mask_batch = episodes_with_prize_mask[indices]
                trigger_episodes_mask_batch = episodes_with_trigger_mask[indices]

                N = observations.shape[0]

                timing = dict()

                with Timing(timing, 'time_preprocess'):
                    observations = observations.transpose(2, 4).transpose(0, 1).to(device)
                    actions = actions.transpose(0, 1).to(device)
                    returns = returns.transpose(0, 1).to(device)

                with Timing(timing, 'time_run_inference'):
                    output, attention_output = model(observations, actions, output_attention=True)

                # Reshape output for K-dimensional CrossEntropy loss
                with Timing(timing, 'time_optimize_model'):
                    output = output.permute(0, 2, 1)
                    returns = returns.transpose(0, 1)

                    # Compute loss
                    optimizer.zero_grad()

                    loss = criterion(output, returns)
                    losses.append(loss.item())

                    loss.backward()

                    # Clip to avoid exploding gradient issues
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

                    optimizer.step()

                with Timing(timing, 'time_evaluate_model'):
                    model.eval()

                    val_loss, val_acc, val_ca_precision, \
                        val_ca_recall, val_rel_attention_vals_avg = evaluate(model, criterion, valid_loader,
                                                                             device, timing, ca_gt)

                    preds = output.argmax(dim=1)

                    padding_mask = actions.transpose(0, 1)[:, :, 0] != PAD_VAL
                    masked_preds = preds[padding_mask]
                    masked_returns = returns[padding_mask]
                    acc = torch.sum(masked_preds == masked_returns) / masked_returns.numel()

                    # Compute credit assignment average graph #

                    # extract attention values *when getting prize*
                    # for some weird reason, attention_matrices[:, prize_timesteps_batch] results in [N, N, S] rather than [N, S]
                    # fixed with help of:
                    # https://discuss.pytorch.org/t/selecting-element-on-dimension-from-list-of-indexes/36319/2?u=nick-baliesnyi
                    attention_vals = attention_output[torch.arange(N), prize_timesteps_batch]

                    # zero out episodes where prize wasn't touched
                    attention_vals *= prize_episodes_mask_batch[:, None]

                    # compute relative timesteps for each episode relative to trigger activation
                    # e.g.
                    # [[-10, -9, -8, ..., 39],
                    # [-25, -24, -24,..., 24],
                    #                     ...]
                    rel_timesteps = torch.arange(0, seq_len) + seq_len
                    rel_timesteps = rel_timesteps.repeat(
                        N, 1).to(device)
                    rel_timesteps -= trigger_timesteps_batch[:, None]

                    # scatter the batch's attention values over a relative timestep matrix (N, S*2+1)
                    n_relevant_episodes = torch.sum(torch.logical_and(prize_episodes_mask_batch, trigger_episodes_mask_batch))
                    rel_attention_vals_batch = torch.zeros_like(rel_attention_vals).scatter(1, rel_timesteps, attention_vals) / n_relevant_episodes
                    rel_attention_vals += rel_attention_vals_batch

                    # Compute credit assignment precision/recall #

                    attention_discrete = torch.gt(attention_vals, attention_threshold, out=torch.empty(
                        attention_vals.shape, dtype=torch.uint8, device=device))
                    true_positives = torch.sum(torch.logical_and(
                        attention_discrete, trigger_activations_batch))

                    ca_precision = true_positives / torch.sum(attention_discrete)
                    ca_recall = true_positives / n_relevant_episodes

                    rel_attention_vals_avg = torch.sum(rel_attention_vals, axis=0)
                    rel_attention_vals *= 0

                with Timing(timing, 'time_draw_ca_plots'):
                    # create credit assignment plots
                    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6))

                    x_axis = (torch.arange(rel_attention_vals_avg.shape[0]) - seq_len).cpu()

                    axes[0].set_title('Training (single batch)')
                    axes[0].set_xlabel('Relative step in trajectory (0 - activating trigger)')
                    axes[0].set_ylabel('Average attention weights')
                    axes[0].plot(x_axis, rel_attention_vals_avg.cpu())

                    axes[1].set_title('Validation (average across batches)')
                    axes[1].set_xlabel('Relative step in trajectory (0 - activating trigger)')
                    axes[1].set_ylabel('Average attention weights')
                    axes[1].plot(x_axis, val_rel_attention_vals_avg.cpu())

                    wandb_plot_image = wandb.Image(plt)

                if val_ca_precision >= best_val_precision and \
                        val_ca_recall >= best_val_recall and \
                        val_acc >= best_val_acc and \
                        val_loss <= best_val_loss:
                    with Timing(timing, 'time_save_model'):
                        best_val_precision = val_ca_precision
                        best_val_recall = val_ca_recall
                        best_val_acc = val_acc
                        best_val_loss = val_loss
                        torch.save(model, os.path.join(
                            save_dir, f'model_r{val_ca_recall:.3f}_p{val_ca_precision:.3f}_a{val_acc:.3f}_l{val_loss:.3f}.pt'))

                wandb.log({
                    'average_attention': wandb_plot_image,
                    'acc': acc,
                    'ca_precision': ca_precision,
                    'ca_recall': ca_recall,
                    'epoch': epoch,
                    'loss': loss,
                    'val_acc': val_acc,
                    'val_ca_presicion': val_ca_precision,
                    'val_ca_recall': val_ca_recall,
                    'val_loss': val_loss,
                    **{k: v['time'] / v['count'] for k, v in timing.items()}
                }, )
                plt.close()
                timing = dict()

                pbar.set_postfix_str(
                    f'''loss: {loss:.3f}, acc: {acc:.3f}, val_loss: {val_loss:.3f}, val_acc: {val_acc:.3f}, 
                        val_ca_p: {val_ca_precision:.3f}, val_ca_r: {val_ca_recall:.3f}''')

                model.train()

                del observations
                del actions
                del returns
                del output
                del preds

            # free some GPU memory
            torch.cuda.empty_cache()

            mean_loss = sum(losses) / len(losses)
            scheduler.step(mean_loss)

        torch.save(model, os.path.join(save_dir, 'final_model.pt'))
        wandb.finish()


if __name__ == '__main__':
    train()

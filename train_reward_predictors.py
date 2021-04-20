import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import click
import os
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import wandb
from timing import Timing
from tqdm import tqdm
import matplotlib.pyplot as plt

from reward_predictor import RewardPredictor
from trajectories import preprocess_dataset, find_activations
from config import PAD_VAL

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_ca_evaluation(model_path, data_path,
                      observations_mean_path, observations_std_path,
                      wandb,
                      attention_threshold,
                      batch_size,
                      ca_ignore_episodes_without_triggers,
                      use_wandb=True,
                      use_returns=False):

    # 1. Load and pre-process data
    dataset = torch.load(data_path)
    dataset_size = dataset.observations.shape[0]

    # global variables used in other parts
    seq_len = dataset.observations.shape[1]
    criterion = nn.CrossEntropyLoss(ignore_index=int(PAD_VAL))  # will be ignored
    timing = dict()  # will be ignored

    # credit assignment ground truths
    # has to be done *before* pre-processing, as it relies on non-normalized pixel observations
    trigger_activations = find_activations(dataset.observations, dataset.actions, target='trigger').to(device)  # N, S
    prize_activations = find_activations(dataset.observations, dataset.actions, target='prize')  # N, S

    trigger_activations_indices = torch.argmax(trigger_activations, axis=-1).to(device)  # (N,) timesteps preceding trigger activation
    prize_activations_indices = torch.argmax(prize_activations, axis=-1)  # (N,) timesteps preceding taking prize

    episodes_with_prize_mask = torch.sum(prize_activations, axis=-1).to(device)  # mask for episodes where prizes were taken
    episodes_with_trigger_mask = torch.sum(trigger_activations, axis=-1).to(device)  # (N, )

    ca_gt = {
        'attention_threshold': attention_threshold,
        'batch_size': batch_size,
        'episodes_with_trigger_mask': episodes_with_trigger_mask,
        'episodes_with_prize_mask': episodes_with_prize_mask,
        'seq_len': seq_len,
        'trigger_activations': trigger_activations,
        'trigger_timesteps': trigger_activations_indices,
        'prize_activations': prize_activations,
        'prize_timesteps': prize_activations_indices,
    }

    data_loader = preprocess_dataset(dataset, data_path, sum_rewards=use_returns, normalize=True,
                                     eval_mode=True, path_to_mean=observations_mean_path, path_to_std=observations_std_path)

    # 3. Evaluate each model on the data
    model = torch.load(model_path).to(device)
    _, acc, ca_precision, ca_recall, rel_attention_vals, preds, trues = evaluate(model, criterion, data_loader, device,
                                                                                 timing, ca_gt, ca_ignore_episodes_without_triggers)

    # confusion matrix
    c_matrix = confusion_matrix(trues, preds, normalize=None)
    c_matrix_plot = ConfusionMatrixDisplay(c_matrix, display_labels=['Negative', 'Zero', 'Positive'])
    c_matrix_plot.plot(values_format='')
    c_matrix_plot_img = wandb.Image(c_matrix_plot.figure_)
    plt.close('all')

    # attention plot
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    x_axis = (torch.arange(rel_attention_vals.shape[0]) - seq_len).cpu()

    axes.set_title(f'Evaluation ({dataset_size} trajectories)')
    axes.set_xlabel('Relative step in trajectory (0 - activating trigger)')
    axes.set_ylabel('Average attention weights')
    axes.plot(x_axis, rel_attention_vals.cpu())

    # 4. Log results to Wandb

    if use_wandb:
        wandb.log({
            'test_average_attention': wandb.Image(plt),
            'test_acc': acc,
            'test_ca_precision': ca_precision,
            'test_ca_recall': ca_recall,
            'test_confusion_matrix': c_matrix_plot_img,
        })
    else:
        return acc, ca_precision, ca_recall, axes, ca_gt, rel_attention_vals


def evaluate(model, criterion, data_loader, device, timing, ca_gt, ca_skip_episodes_without_triggers, verbose=False):
    attention_threshold, seq_len = ca_gt['attention_threshold'], ca_gt['seq_len']

    losses = []
    accuracies = []
    ca_precisions = []
    ca_recalls = []
    y_preds = []
    y_trues = []
    rel_attention_vals = torch.zeros((seq_len*2+1), device=device)  # S*2+1

    n_attn_plot_batches = 0

    for batch_idx, batch in enumerate(data_loader):
        observations, actions, returns, indices = batch

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

            # balanced accuracy
            acc_neg = torch.sum(torch.logical_and(masked_preds == 0, masked_returns == 0)) / torch.sum(masked_returns == 0)
            acc_zero = torch.sum(torch.logical_and(masked_preds == 1, masked_returns == 1)) / torch.sum(masked_returns == 1)
            acc_pos = torch.sum(torch.logical_and(masked_preds == 2, masked_returns == 2)) / torch.sum(masked_returns == 2)
            acc = torch.mean(torch.tensor([acc_neg, acc_zero, acc_pos]))

            # Compute credit assignment average graph #

            # for some weird reason, attention_matrices[:, prize_timesteps_batch] results in [N, N, S] rather than [N, S]
            # fixed with help of:
            # https://discuss.pytorch.org/t/selecting-element-on-dimension-from-list-of-indexes/36319/2?u=nick-baliesnyi
            attention_vals = attention_output[torch.arange(N), prize_timesteps_batch]

            # zero out attention for episodes where prize wasn't touched
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
            # zero out attention when triggers weren't touched
            # then average over the episodes to get (S*2+1)
            n_episodes_prizes_and_triggers = torch.sum(torch.logical_and(prize_episodes_mask_batch, trigger_episodes_mask_batch))
            if n_episodes_prizes_and_triggers > 0:
                n_attn_plot_batches += 1
                rel_attention_vals_batch = torch.zeros((N, seq_len*2+1), device=device).scatter(1, rel_timesteps, attention_vals)
                rel_attention_vals_batch *= trigger_episodes_mask_batch[:, None]
                rel_attention_vals_batch = torch.sum(rel_attention_vals_batch, axis=0) / n_episodes_prizes_and_triggers  # (S*2+1)
                rel_attention_vals += rel_attention_vals_batch

            # Compute credit assignment precision/recall #
            if ca_skip_episodes_without_triggers:
                attention_vals *= trigger_episodes_mask_batch[:, None]

            attention_discrete = torch.gt(attention_vals, attention_threshold, out=torch.empty(
                                          attention_vals.shape, dtype=torch.uint8, device=device))
            true_positives = torch.sum(torch.logical_and(attention_discrete, trigger_activations_batch))

            if torch.sum(attention_discrete) > 0:
                ca_precision = true_positives / torch.sum(attention_discrete)
                ca_precisions.append(ca_precision.item())

            if n_episodes_prizes_and_triggers > 0:
                ca_recall = true_positives / n_episodes_prizes_and_triggers
                ca_recalls.append(ca_recall.item())

            losses.append(loss.item())
            accuracies.append(acc.item())

        del observations
        del actions
        del returns
        del output

    mean_loss = sum(losses) / len(losses)
    mean_acc = sum(accuracies) / len(accuracies)
    mean_ca_precision = sum(ca_precisions) / len(ca_precisions)
    mean_ca_recall = sum(ca_recalls) / len(ca_recalls)
    mean_rel_attention_vals = rel_attention_vals / n_attn_plot_batches

    del losses
    del accuracies
    del ca_precisions
    del ca_recalls

    if verbose:
        print('mean_loss:', mean_loss)
        print('mean_acc:', mean_acc)

    return mean_loss, mean_acc, mean_ca_precision, mean_ca_recall, mean_rel_attention_vals, y_preds, y_trues


@click.command()
@click.option('--agent', type=str, required=True)
@click.option('--group', type=str, required=True)
@click.option('--note', type=str, help='message to explain how is this run different')
@click.option('--data-path', type=click.Path(exists=True), help='path to trajectories dataset', required=True)
@click.option('--test-path', type=click.Path(exists=True), help='path to test trajectories dataset', required=True)
@click.option('--seed', type=int, default=42, help='random seed used')
@click.option('--seeds', type=int, default=1, help='# of random seeds used')
@click.option('--log-frequency', type=int, default=5e1, help='logging frequency, iterations')
@click.option('--learning-rate', type=float, default=3e-3, help='goal learning rate')
@click.option('--epochs', type=int, default=10, help='number of epochs to train for')
@click.option('--batch-size', default=128, help='training batch size')
@click.option('--use-returns/--use-rewards', default=False, help='whether to predict returns instead of single-step rewards')
@click.option('--attention-threshold', default=0.2, help='threshold attention weight discretization')
@click.option('--skip-no-trigger-episodes-precision/--include-no-trigger-episodes-precision', required=True,
              help='whether to skip no-trigger episodes when calculating credit assignment precision')
@click.option('--valid-size', type=float, default=0.2, help='proportion of validation set')
@click.option('--use-wandb/--no-wandb', default=True)
def train(agent, group, note, data_path, test_path, seed, seeds, log_frequency,
          learning_rate, epochs, batch_size, use_returns,
          attention_threshold, skip_no_trigger_episodes_precision, valid_size, use_wandb):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    wandb.login()

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

    # credit assignment ground truths
    ca_gt = {
        'attention_threshold': attention_threshold,
        'episodes_with_trigger_mask': episodes_with_trigger_mask,
        'episodes_with_prize_mask': episodes_with_prize_mask,
        'trigger_activations': trigger_activations,
        'trigger_timesteps': trigger_timesteps,
        'prize_activations': prize_activations,
        'prize_timesteps': prize_timesteps,
        'seq_len': seq_len,
    }

    train_loader, valid_loader, obs_mean_path, obs_std_path = preprocess_dataset(dataset, data_path,
                                                                                 sum_rewards=use_returns,
                                                                                 batch_size=batch_size,
                                                                                 valid_size=valid_size,
                                                                                 seed=seed)

    # model
    class_weights = torch.tensor([0.499, 0.02, 0.499]).to(device)

    for i_seed in range(seed, seed+seeds):

        wandb.init(project='attentional_fw_baselines',
                   entity='ut-rl-credit',
                   notes='Reward predictor',
                   group=group,
                   tags=['SECRET', 'verification'],
                   mode='online' if use_wandb else 'disabled',
                   config=dict(
                       agent=agent,
                       attention_threshold=attention_threshold,
                       batch_size=batch_size,
                       data_path=data_path,
                       test_path=test_path,
                       epochs=epochs,
                       learning_rate=learning_rate,
                       seed=i_seed,
                       valid_size=valid_size,
                       use_returns=use_returns,
                       skip_no_trigger_episodes_precision=skip_no_trigger_episodes_precision,
                   ))
        # Upload models at the end of training
        save_dir = wandb.run.dir if use_wandb else './'
        BEST_MODEL_PATH = os.path.join(save_dir, 'best_model.pt')
        FINAL_MODEL_PATH = os.path.join(save_dir, 'final_model.pt')
        wandb.save(os.path.join(save_dir, "*.pt"))

        torch.cuda.manual_seed(i_seed)
        torch.manual_seed(i_seed)
        np.random.seed(i_seed)
        random.seed(i_seed)

        obs_space_dims = [dataset.observations.shape[-1], *dataset.observations.shape[2:-1]]
        act_space_size = dataset.actions.shape[-1]
        model = RewardPredictor(obs_space_dims, act_space_size, device, verbose=False).to(device)

        wandb.watch(model)

        best_val_precision = 0
        best_val_recall = 0
        best_val_acc = 0
        best_val_loss = 10000

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=10, verbose=True
        )

        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=int(PAD_VAL))

        print(f'Training reward predictor seed={i_seed}...\n')

        for epoch in tqdm(range(epochs)):

            losses = []

            for batch_idx, batch in enumerate(train_loader):
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
                        val_ca_recall, val_rel_attention_vals_avg, \
                        _, _ = evaluate(model, criterion, valid_loader,
                                        device, timing, ca_gt, skip_no_trigger_episodes_precision)

                    preds = output.argmax(dim=1)

                    padding_mask = actions.transpose(0, 1)[:, :, 0] != PAD_VAL
                    masked_preds = preds[padding_mask]
                    masked_returns = returns[padding_mask]

                    # balanced accuracy
                    acc_neg = torch.sum(torch.logical_and(masked_preds == 0, masked_returns == 0)) / torch.sum(masked_returns == 0)
                    acc_zero = torch.sum(torch.logical_and(masked_preds == 1, masked_returns == 1)) / torch.sum(masked_returns == 1)
                    acc_pos = torch.sum(torch.logical_and(masked_preds == 2, masked_returns == 2)) / torch.sum(masked_returns == 2)
                    acc = torch.mean(torch.tensor([acc_neg, acc_zero, acc_pos]))

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
                    n_episodes_prizes_and_triggers = torch.sum(torch.logical_and(prize_episodes_mask_batch, trigger_episodes_mask_batch))
                    if n_episodes_prizes_and_triggers > 0:
                        rel_attention_vals_batch = torch.zeros((N, seq_len*2+1), device=device).scatter(1, rel_timesteps, attention_vals)
                        rel_attention_vals_batch *= trigger_episodes_mask_batch[:, None]
                        rel_attention_vals_batch = torch.sum(rel_attention_vals_batch, axis=0) / n_episodes_prizes_and_triggers  # (S*2+1)

                    # Compute credit assignment precision/recall #
                    if skip_no_trigger_episodes_precision:
                        attention_vals *= trigger_episodes_mask_batch[:, None]

                    attention_discrete = torch.gt(attention_vals, attention_threshold, out=torch.empty(
                                                  attention_vals.shape, dtype=torch.uint8, device=device))
                    true_positives = torch.sum(torch.logical_and(
                                               attention_discrete, trigger_activations_batch))

                    ca_precision = true_positives / torch.sum(attention_discrete)
                    ca_recall = true_positives / n_episodes_prizes_and_triggers

                with Timing(timing, 'time_draw_ca_plots'):

                    # create credit assignment plots
                    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6))

                    x_axis = (torch.arange(rel_attention_vals_batch.shape[0]) - seq_len).cpu()

                    axes[0].set_title('Training (single batch)')
                    axes[0].set_xlabel('Relative step in trajectory (0 - activating trigger)')
                    axes[0].set_ylabel('Average attention weights')
                    axes[0].plot(x_axis, rel_attention_vals_batch.cpu())

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
                        torch.save(model, BEST_MODEL_PATH)

                log_dict = {
                    'average_attention': wandb_plot_image,
                    'acc': acc,
                    'epoch': epoch,
                    'loss': loss,
                    'val_acc': val_acc,
                    'val_ca_precision': val_ca_precision,
                    'val_ca_recall': val_ca_recall,
                    'val_loss': val_loss,
                    **{k: v['time'] / v['count'] for k, v in timing.items()}
                }
                if torch.sum(attention_discrete) > 0:
                    log_dict['ca_precision'] = ca_precision
                if n_episodes_prizes_and_triggers > 0:
                    log_dict['ca_recall'] = ca_recall

                wandb.log(log_dict)
                plt.close('all')
                timing = dict()

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

        torch.save(model, FINAL_MODEL_PATH)
        del model

        print(f'Evaluating reward predictor seed={i_seed}...\n')
        # Evaluate best model on the test set and log results
        run_ca_evaluation(BEST_MODEL_PATH, test_path, obs_mean_path, obs_std_path, wandb, attention_threshold,
                          batch_size, skip_no_trigger_episodes_precision, use_returns=use_returns)

        wandb.finish()


if __name__ == '__main__':
    train()

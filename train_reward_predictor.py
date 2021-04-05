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


def evaluate(model, criterion, data_loader, device, timing, verbose=False):
    losses = []
    accuracies = []

    for batch_idx, batch in enumerate(data_loader):
        observations, actions, returns, indices = batch

        with Timing(timing, 'time_eval_preprocess'):
            observations = observations.transpose(
                2, 4).transpose(0, 1).to(device)
            actions = actions.transpose(0, 1).to(device)
            returns = returns.transpose(0, 1).to(device)

        with Timing(timing, 'time_eval_run_inference'):
            output = model(observations, actions)

        with Timing(timing, 'time_eval_calc_metrics'):
            # reshape for CrossEntropyLoss
            output = output.permute(0, 2, 1)
            returns = returns.transpose(0, 1)

            loss = criterion(output, returns)
            preds = output.argmax(dim=1)
            masked_preds = preds[returns != PAD_VAL]
            masked_returns = returns[returns != PAD_VAL]
            accuracy = torch.sum(masked_preds == masked_returns) / \
                masked_returns.numel()

        losses.append(loss.item())
        accuracies.append(accuracy.item())

        del observations
        del actions
        del returns
        del output

    mean_loss = sum(losses) / len(losses)
    mean_acc = sum(accuracies) / len(accuracies)

    del losses
    del accuracies

    if verbose:
        print('mean_loss:', mean_loss)
        print('mean_acc:', mean_acc)

    return mean_loss, mean_acc


@click.command()
@click.option('--note', type=str, help='message to explain how is this run different', required=True)
@click.option('--data-path', type=click.Path(exists=True), help='path to trajectories dataset', required=True)
@click.option('--seed', type=int, default=42, help='random seed used')
@click.option('--log-frequency', type=int, default=5e1, help='logging frequency, iterations')
@click.option('--learning-rate', type=float, default=3e-3, help='goal learning rate')
@click.option('--epochs', type=int, default=10, help='number of epochs to train for')
@click.option('--batch-size', default=128, help='training batch size')
@click.option('--attention-threshold', default=0.2, help='threshold attention weight discretization')
@click.option('--valid-size', type=float, default=0.2, help='proportion of validation set')
@click.option('--use-wandb/--no-wandb', default=True)
def train(note, data_path, seed, log_frequency,
          learning_rate, epochs, batch_size, 
          attention_threshold, valid_size, use_wandb):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    wandb.login()
    wandb.init(project='attentional_fw_baselines',
               entity='ut-rl-credit',
               notes=note,
               mode='online' if use_wandb else 'disabled',
               config=dict(
                   attention_threshold=attention_threshold,
                   batch_size=batch_size,
                   data_path=data_path,
                   epochs=epochs,
                   learning_rate=learning_rate,
                   seed=seed,
                   valid_size=valid_size,
               ))
    # Upload models at the end of training
    save_dir = wandb.run.dir if use_wandb else './'
    wandb.save(os.path.join(save_dir, "*.pt"))

    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data
    dataset = torch.load(data_path)
    trigger_activations = find_activations(
        dataset.observations, dataset.actions, target='trigger').to(device)
    # indices of steps preceding activating trigger
    trigger_activations_indices = torch.argmax(
        trigger_activations, axis=-1).to(device)
    prize_activations = find_activations(
        dataset.observations, dataset.actions, target='prize')
    # indices of steps preceding taking prize
    prize_activations_indices = torch.argmax(prize_activations, axis=-1)
    # mask for episodes where prizes were taken
    episodes_with_trigger_mask = torch.sum(trigger_activations, axis=-1).to(device)
    episodes_with_prize_mask = torch.sum(prize_activations, axis=-1).to(device)
    train_loader, valid_loader = preprocess_dataset(
        dataset, data_path, batch_size=batch_size, valid_size=valid_size, seed=seed)

    seq_len = dataset.observations.shape[1]
    attention_weights_global = torch.zeros(
        (batch_size, seq_len*2+1), device=device)  # N, S*2

    # model
    class_weights = torch.tensor([0.499, 0.02, 0.499]).to(device)
    model = RewardPredictor(
        OBSERVATION_SPACE_DIMS, ACTION_SPACE_SIZE, device, verbose=False).to(device)

    wandb.watch(model)

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

            N = observations.shape[0]

            timing = dict()

            with Timing(timing, 'time_preprocess'):
                observations = observations.transpose(
                    2, 4).transpose(0, 1).to(device)
                actions = actions.transpose(0, 1).to(device)
                returns = returns.transpose(0, 1).to(device)

            with Timing(timing, 'time_run_inference'):
                output, attention_output = model(
                    observations, actions, output_attention=True)

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
                val_loss, val_acc = evaluate(
                    model, criterion, valid_loader, device, timing)

                preds = output.argmax(dim=1)
                masked_preds = preds[returns != PAD_VAL]
                masked_returns = returns[returns != PAD_VAL]
                acc = torch.sum(masked_preds == masked_returns) / \
                    masked_returns.numel()

                # credit assignment ground truth
                trigger_activations_batch = trigger_activations[indices]
                trigger_activations_indices_batch = trigger_activations_indices[indices]
                # moments at which we want to evaluate attention
                prize_activations_batch = prize_activations_indices[indices]
                # mask to zero out attention in episodes without touching prize
                prize_episodes_mask_batch = episodes_with_prize_mask[indices]
                trigger_episodes_mask_batch = episodes_with_trigger_mask[indices]

                # for some weird reason, attention_matrices[:, prize_activations_batch] results in [N, N, S] rather than [N, S]
                # fixed with help of: https://discuss.pytorch.org/t/selecting-element-on-dimension-
                # from-list-of-indexes/36319/2?u=nick-baliesnyi
                attention_vals = attention_output[torch.arange(
                    N), prize_activations_batch]
                # zero out episodes where prize wasn't touched
                attention_vals *= prize_episodes_mask_batch[:, None]

                attention_indices = torch.arange(0, seq_len) + seq_len
                attention_indices = attention_indices.repeat(
                    N, 1).to(device)
                attention_indices -= trigger_activations_indices_batch[:, None]

                added_values = torch.zeros_like(attention_weights_global)
                attention_weights_global += added_values.scatter(
                    1, attention_indices, attention_vals)

                attention_discrete = torch.gt(attention_vals, attention_threshold, out=torch.empty(
                    attention_vals.shape, dtype=torch.uint8, device=device))
                true_positives = torch.sum(torch.logical_and(
                    attention_discrete, trigger_activations_batch))

                # TODO: evaluate on the whole dataset
                attention_precision = true_positives / torch.sum(attention_discrete)
                N_relevant_recall_episodes = torch.sum(torch.logical_and(prize_episodes_mask_batch, trigger_episodes_mask_batch))

                attention_recall = true_positives / N_relevant_recall_episodes

                print(f'attention paid somewhere in batch (N={N})', torch.sum(attention_discrete))
                print(f'true positives in batch (N={N})', true_positives)
                print(f'trigger_activations in batch (N={N})', torch.sum(trigger_activations_batch))

            weights_averaged = torch.sum(
                attention_weights_global, axis=0) / (N)
            attention_weights_global *= 0
            x_axis = torch.arange(weights_averaged.shape[0]) - seq_len
            plt.xlabel('Relative step in trajectory (0 - activating trigger)')
            plt.ylabel('Average attention weights')
            plt.plot(x_axis.cpu(), weights_averaged.cpu())

            wandb.log({
                'loss': loss,
                'acc': acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'epoch': epoch,
                'average_attention': plt,
                'ca_precision': attention_precision,
                'ca_recall': attention_recall,
                **{k: v['time'] / v['count'] for k, v in timing.items()}
            }, )
            plt.cla()
            timing = dict()

            pbar.set_postfix_str(
                f'loss: {loss:0.5f}, acc: {acc:0.5f}, val_loss: {val_loss:0.5f}, val_acc: {val_acc:0.5f}')

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

    torch.save(model, os.path.join(save_dir, 'model.pt'))
    wandb.finish()


if __name__ == '__main__':
    train()

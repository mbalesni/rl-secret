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

from reward_predictor import RewardPredictor
from trajectories import load_trajectories, get_data_loaders

PAD_VAL = 10
ACTION_SPACE_SIZE = 3
OBSERVATION_SPACE_DIMS = (3, 24, 24)


def evaluate(model, criterion, data_loader, device, timing, verbose=False):
    losses = []
    accuracies = []

    for batch_idx, batch in enumerate(data_loader):
        observations, actions, rewards = batch

        with Timing(timing, 'time_eval_preprocess'):
            observations = observations.transpose(
                2, 4).transpose(0, 1).to(device)
            actions = actions.transpose(0, 1).to(device)
            rewards = rewards.transpose(0, 1).to(device)

            returns = torch.ones_like(rewards)
            # return for each episode in batch
            returns += torch.sum(rewards, axis=0)[None, :]
            returns += 1  # turns values (-1,0,1) into "classes" (0,1,2)

        with Timing(timing, 'time_eval_run_inference'):
            output = model(observations, actions)

        with Timing(timing, 'time_eval_calc_metrics'):
            # reshape for CrossEntropyLoss
            output = output.permute(1, 2, 0)
            rewards = rewards.transpose(0, 1)

            loss = criterion(output, rewards)
            preds = output.argmax(dim=1)
            masked_preds = preds[rewards != PAD_VAL]
            masked_rewards = rewards[rewards != PAD_VAL]
            accuracy = torch.sum(masked_preds == masked_rewards) / \
                masked_rewards.numel()

        losses.append(loss.item())
        accuracies.append(accuracy.item())

        del observations
        del actions
        del rewards
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
@click.option('--data', type=click.Path, help='path to trajectories dataset', required=True)
@click.option('--seed', type=int, default=42, help='random seed used')
@click.option('--log-frequency', type=int, default=5e1, help='logging frequency, iterations')
@click.option('--learning-rate', type=float, default=3e-3, help='goal learning rate')
@click.option('--epochs', type=int, default=10, help='number of epochs to train for')
@click.option('--batch-size', default=128, help='training batch size')
@click.option('--valid-size', type=float, default=0.2, help='proportion of validation set')
@click.option('--use-wandb/--no-wandb', default=True)
def train(note, data, seed, log_frequency,
          learning_rate, epochs, batch_size, valid_size, use_wandb):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    wandb.login()
    wandb.init(project='attentional_fw_baselines',
               entity='ut-rl-credit',
               notes=note,
               mode='online' if use_wandb else 'disabled',
               config=dict(
                   seed=seed,
                   learning_rate=learning_rate,
                   batch_size=batch_size,
               ))
    # Upload models at the end of training
    save_dir = wandb.run.dir if use_wandb else './'
    wandb.save(os.path.join(save_dir, "*.pt"))

    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # data
    dataset = load_trajectories(data)
    train_loader, valid_loader = get_data_loaders(
        dataset, batch_size, valid_size)

    # model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_weights = torch.tensor([0.499, 0.02, 0.499]).to(device)
    model = RewardPredictor(
        OBSERVATION_SPACE_DIMS, ACTION_SPACE_SIZE, device, verbose=False).to(device)

    wandb.watch()

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
            observations, actions, rewards = batch

            timing = dict()

            with Timing(timing, 'time_preprocess'):
                observations = observations.transpose(
                    2, 4).transpose(0, 1).to(device)
                actions = actions.transpose(0, 1).to(device)
                rewards = rewards.transpose(0, 1).to(device)

                returns = torch.ones_like(rewards)
                # return for each episode in batch
                returns += torch.sum(rewards, axis=0)[None, :]
                returns += 1  # turns values (-1,0,1) into "classes" (0,1,2)

            with Timing(timing, 'time_run_inference'):
                output = model(observations, actions)

            # Reshape output for K-dimensional CrossEntropy loss
            with Timing(timing, 'time_optimize_model'):
                output = output.permute(1, 2, 0)
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
                masked_preds = preds[rewards != PAD_VAL]
                masked_rewards = rewards[rewards != PAD_VAL]
                acc = torch.sum(masked_preds == masked_rewards) / \
                    masked_rewards.numel()

            wandb.log({
                'loss': loss,
                'acc': acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'epoch': epoch,
                **{k: v['time'] / v['count'] for k, v in timing.items()}
            }, )
            timing = dict()

            pbar.set_postfix_str(
                f'loss: {loss:0.5f}, acc: {acc:0.5f}, val_loss: {val_loss:0.5f}, val_acc: {val_acc:0.5f}')

            model.train()

            del observations
            del actions
            del rewards
            del output
            del preds

        # free some GPU memory
        torch.cuda.empty_cache()

        mean_loss = sum(losses) / len(losses)
        scheduler.step(mean_loss)

    wandb.finish()


if __name__ == '__main__':
    train()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import click
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.interpolate import interp1d

import os
import sys
import gc
import wandb
from tqdm import tqdm

from secret.src.timing import Timing
from secret.src.reward_predictor import RewardPredictor
from secret.envs.triggers.trajectories import preprocess_dataset, find_activations
from secret.src.config import PAD_VAL

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trajectories_module_path = '../envs/triggers'
if trajectories_module_path not in sys.path:
    sys.path.append(trajectories_module_path)


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

    trigger_timesteps_list = [torch.nonzero(act).reshape(-1) for act in trigger_activations]
    filler = np.arange(0, seq_len)
    trigger_timesteps = torch.empty_like(trigger_activations)
    for i, episode_triggers in enumerate(trigger_timesteps_list):

        if len(episode_triggers) > 1:
            # several triggers

            closest_triggers = interpolator(episode_triggers.cpu(), filler)  # fill with 0s if no triggers?
            closest_triggers = torch.tensor(closest_triggers, dtype=torch.int8, device=device)
        elif len(episode_triggers) == 1:
            # single trigger

            closest_triggers = torch.zeros((seq_len), dtype=torch.int8, device=device) + episode_triggers
        else:
            # no triggers

            closest_triggers = torch.zeros((seq_len), dtype=torch.int8, device=device)

        trigger_timesteps[i] = closest_triggers

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
        'trigger_timesteps': trigger_timesteps,
        'prize_activations': prize_activations,
        'prize_timesteps': prize_activations_indices,
    }

    data_loader = preprocess_dataset(dataset, data_path, sum_rewards=use_returns, normalize=True,
                                     eval_mode=True, path_to_mean=observations_mean_path, path_to_std=observations_std_path)

    # 3. Evaluate each model on the data
    model = torch.load(model_path).to(device)
    model.eval()
    with torch.no_grad():
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


def forward_batch(model, criterion, batch, device, timing, ca_gt, skip_no_trigger_episodes_precision):
    observations, actions, returns, indices = batch

    with Timing(timing, 'time_preprocess'):
        attention_threshold = ca_gt['attention_threshold']
        trigger_activations_batch = ca_gt['trigger_activations'][indices]
        trigger_timesteps_batch = ca_gt['trigger_timesteps'][indices]
        prize_timesteps_batch = ca_gt['prize_timesteps'][indices]
        prize_episodes_mask_batch = ca_gt['episodes_with_prize_mask'][indices]
        trigger_episodes_mask_batch = ca_gt['episodes_with_trigger_mask'][indices]

        N, seq_len = observations.shape[:2]

        observations = observations.transpose(2, 4).transpose(0, 1).to(device)
        actions = actions.transpose(0, 1).to(device)
        returns = returns.transpose(0, 1).to(device)

    with Timing(timing, 'time_run_inference'):
        output, attention_output = model(observations, actions, output_attention=True)
        output = output.permute(0, 2, 1)
        returns = returns.transpose(0, 1)

    with Timing(timing, 'time_calc_batch_metrics'):
        loss = criterion(output, returns)
        preds = output.argmax(dim=1)

        padding_mask = actions.transpose(0, 1)[:, :, 0] != PAD_VAL
        masked_preds = preds[padding_mask]
        masked_returns = returns[padding_mask]

        # balanced accuracy
        n_neg_rewards = torch.sum(masked_returns == 0)
        n_zero_rewards = torch.sum(masked_returns == 1)
        n_pos_rewards = torch.sum(masked_returns == 2)

        accs = []
        if n_neg_rewards > 0:
            acc_neg = (torch.sum(torch.logical_and(masked_preds == 0, masked_returns == 0)) / n_neg_rewards)
            accs.append(acc_neg)
        if n_zero_rewards > 0:
            acc_zero = (torch.sum(torch.logical_and(masked_preds == 1, masked_returns == 1)) / n_zero_rewards)
            accs.append(acc_zero)
        if n_pos_rewards > 0:
            acc_pos = (torch.sum(torch.logical_and(masked_preds == 2, masked_returns == 2)) / n_pos_rewards)
            accs.append(acc_pos)
        acc = torch.mean(torch.tensor(accs))

        # Compute credit assignment average graph #

        # extract attention values *when getting prize*
        # for some weird reason, attention_matrices[:, prize_timesteps_batch] results in [N, N, S] rather than [N, S]
        # fixed with help of:
        # https://discuss.pytorch.org/t/selecting-element-on-dimension-from-list-of-indexes/36319/2?u=nick-baliesnyi
        attention_vals = attention_output[torch.arange(N), prize_timesteps_batch]

        # zero out episodes where prize wasn't touched
        attention_vals *= prize_episodes_mask_batch[:, None]

        # compute relative timesteps for each episode relative to trigger activation
        # e.g. if trigger = 10
        # rel_timesteps = [-10, -9, -8, ..., 39]
        #
        # also works for several triggers, e.g. if triggers == [6, 7, 17]
        # rel_timesteps = (without adding `seq_len`):
        # [-6, -5, -4, -3, -2, -1,  0,  0,  1,  2,  3,  4,  5, -4, -3, -2, -1,  0,
        #   1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        #   19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        #
        rel_timesteps = torch.arange(0, seq_len) + seq_len
        rel_timesteps = rel_timesteps.repeat(N, 1).to(device)
        rel_timesteps -= trigger_timesteps_batch

        # scatter the batch's attention values over a relative timestep matrix (N, S*2+1)
        n_episodes_prizes_and_triggers = torch.sum(torch.logical_and(prize_episodes_mask_batch, trigger_episodes_mask_batch))
        rel_attention_vals = torch.zeros((N, seq_len*2+1), device=device)
        if n_episodes_prizes_and_triggers > 0:
            rel_attention_vals.scatter_(1, rel_timesteps, attention_vals)
            rel_attention_vals *= trigger_episodes_mask_batch[:, None]
            rel_attention_vals = torch.sum(rel_attention_vals, axis=0) / n_episodes_prizes_and_triggers  # (S*2+1)

        # Compute credit assignment precision/recall #
        if skip_no_trigger_episodes_precision:
            attention_vals *= trigger_episodes_mask_batch[:, None]

        attention_discrete = torch.gt(attention_vals, attention_threshold, out=torch.empty(
            attention_vals.shape, dtype=torch.uint8, device=device))
        true_positives = torch.sum(torch.logical_and(
            attention_discrete, trigger_activations_batch))

        ca_precision = true_positives / torch.sum(attention_discrete)
        ca_recall = true_positives / n_episodes_prizes_and_triggers

        if torch.sum(attention_discrete) == 0:
            ca_precision = None
        if n_episodes_prizes_and_triggers == 0:
            ca_recall = None

    del observations
    del actions
    del returns
    del output
    del preds

    return loss, acc, ca_precision, ca_recall, rel_attention_vals, masked_preds, masked_returns


def evaluate(model, criterion, data_loader, device, timing, ca_gt, ca_skip_episodes_without_triggers, verbose=False):
    seq_len = ca_gt['seq_len']

    losses = []
    accuracies = []
    ca_precisions = []
    ca_recalls = []
    y_preds = []
    y_trues = []
    n_attn_plot_batches = 0
    rel_attention_vals = torch.zeros((seq_len*2+1), device=device)  # S*2+1

    for batch in data_loader:
        loss_batch, acc_batch, ca_precision_batch, ca_recall_batch, \
            attention_batch, y_preds_batch, y_trues_batch = forward_batch(model, criterion, batch, device, timing,
                                                                          ca_gt, ca_skip_episodes_without_triggers)

        if ca_precision_batch is not None:
            ca_precisions.append(ca_precision_batch.item())

        if ca_recall_batch is not None:
            ca_recalls.append(ca_recall_batch.item())
            n_attn_plot_batches += 1
            rel_attention_vals += attention_batch

        losses.append(loss_batch.item())
        accuracies.append(acc_batch.item())
        y_preds += list(y_preds_batch.cpu())
        y_trues += list(y_trues_batch.cpu())

    mean_loss = sum(losses) / len(losses)
    mean_acc = sum(accuracies) / len(accuracies)
    mean_ca_precision = sum(ca_precisions) / len(ca_precisions)
    mean_ca_recall = sum(ca_recalls) / len(ca_recalls)
    mean_rel_attention_vals = rel_attention_vals / n_attn_plot_batches

    del losses
    del accuracies
    del ca_precisions
    del ca_recalls

    return mean_loss, mean_acc, mean_ca_precision, mean_ca_recall, mean_rel_attention_vals, y_preds, y_trues


def interpolator(values, filler):
    return interp1d(values, values, kind='nearest', fill_value='extrapolate')(filler)


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
@click.option('--skip-no-trigger-episodes-precision/--include-no-trigger-episodes-precision', default=None, required=True,
              help='whether to skip no-trigger episodes when calculating credit assignment precision')
@click.option('--valid-size', type=float, default=0.2, help='proportion of validation set')
@click.option('--use-wandb/--no-wandb', default=True)
def train(agent, group, note, data_path, test_path, seed, seeds, log_frequency,
          learning_rate, epochs, batch_size, use_returns,
          attention_threshold, skip_no_trigger_episodes_precision, valid_size, use_wandb):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    wandb.login()

    for i_seed in range(seed, seed+seeds):

        # data
        dataset = torch.load(data_path)
        seq_len = dataset.observations.shape[1]

        trigger_activations = find_activations(dataset.observations, dataset.actions, target='trigger').to(device)  # (N, S)
        prize_activations = find_activations(dataset.observations, dataset.actions, target='prize')  # (N, S)

        trigger_timesteps_list = [torch.nonzero(act).reshape(-1) for act in trigger_activations]
        filler = np.arange(0, seq_len)
        trigger_timesteps = torch.empty_like(trigger_activations)
        for i, episode_triggers in enumerate(trigger_timesteps_list):

            if len(episode_triggers) > 1:
                # several triggers

                closest_triggers = interpolator(episode_triggers.cpu(), filler)  # fill with 0s if no triggers?
                closest_triggers = torch.tensor(closest_triggers, dtype=torch.int8, device=device)
            elif len(episode_triggers) == 1:
                # single trigger

                closest_triggers = torch.zeros((seq_len), dtype=torch.int8, device=device) + episode_triggers
            else:
                # no triggers

                closest_triggers = torch.zeros((seq_len), dtype=torch.int8, device=device)

            trigger_timesteps[i] = closest_triggers

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

            for batch in train_loader:

                timing = dict()

                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss_batch, acc_batch, ca_precision_batch, ca_recall_batch, \
                    attention_batch, _, _ = forward_batch(model, criterion, batch, device,
                                                          timing, ca_gt, skip_no_trigger_episodes_precision)
                losses.append(loss_batch)

                # Reshape output for K-dimensional CrossEntropy loss
                with Timing(timing, 'time_optimize_model'):
                    loss_batch.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                    optimizer.step()

                with Timing(timing, 'time_evaluate_model'):
                    model.eval()

                    val_loss, val_acc, val_ca_precision, \
                        val_ca_recall, val_rel_attention_vals_avg, \
                        _, _ = evaluate(model, criterion, valid_loader,
                                        device, timing, ca_gt, skip_no_trigger_episodes_precision)

                with Timing(timing, 'time_save_model'):
                    if val_ca_precision >= best_val_precision and \
                            val_ca_recall >= best_val_recall and \
                            val_acc >= best_val_acc and \
                            val_loss <= best_val_loss:
                        best_val_precision = val_ca_precision
                        best_val_recall = val_ca_recall
                        best_val_acc = val_acc
                        best_val_loss = val_loss
                        torch.save(model, BEST_MODEL_PATH)

                log_dict = {
                    'acc': acc_batch,
                    'epoch': epoch,
                    'loss': loss_batch,
                    'val_acc': val_acc,
                    'val_ca_precision': val_ca_precision,
                    'val_ca_recall': val_ca_recall,
                    'val_loss': val_loss,
                    **{k: v['time'] / v['count'] for k, v in timing.items()}
                }
                if ca_precision_batch is not None:
                    log_dict['ca_precision'] = ca_precision_batch
                if ca_recall_batch is not None:
                    log_dict['ca_recall'] = ca_recall_batch
                if torch.count_nonzero(attention_batch) > 0:
                    with Timing(timing, 'time_draw_ca_plots'):
                        # create credit assignment plots
                        fig, axes = plt.subplots(1, 2, figsize=(12.5, 6))

                        x_axis = (torch.arange(attention_batch.shape[0]) - seq_len).cpu()

                        axes[0].set_title('Training (single batch)')
                        axes[0].set_xlabel('Relative step in trajectory (0 - activating trigger)')
                        axes[0].set_ylabel('Average attention weights')
                        axes[0].plot(x_axis, attention_batch.cpu())

                        axes[1].set_title('Validation (average across batches)')
                        axes[1].set_xlabel('Relative step in trajectory (0 - activating trigger)')
                        axes[1].set_ylabel('Average attention weights')
                        axes[1].plot(x_axis, val_rel_attention_vals_avg.cpu())

                        log_dict['average_attention'] = wandb.Image(plt)
                        plt.close('all')

                wandb.log(log_dict)

                # free some GPU memory
                torch.cuda.empty_cache()

            mean_loss = torch.mean(torch.tensor(losses))
            scheduler.step(mean_loss)

        torch.save(model, FINAL_MODEL_PATH)
        del trigger_activations
        del prize_activations
        del trigger_timesteps
        del prize_timesteps
        del episodes_with_trigger_mask
        del episodes_with_prize_mask
        del ca_gt
        del model
        del dataset
        del train_loader
        del valid_loader
        del class_weights
        gc.collect()
        torch.cuda.empty_cache()

        print(f'Evaluating reward predictor seed={i_seed}...\n')
        # Evaluate best model on the test set and log results
        run_ca_evaluation(BEST_MODEL_PATH, test_path, obs_mean_path, obs_std_path, wandb, attention_threshold,
                          batch_size, skip_no_trigger_episodes_precision, use_returns=use_returns, use_wandb=use_wandb)

        wandb.finish()


if __name__ == '__main__':
    train()

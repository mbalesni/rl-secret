import os
from train_reward_predictor import evaluate, PAD_VAL
from trajectories import preprocess_dataset, find_activations
from matplotlib import pyplot as plt
import torch.nn as nn
import torch
import wandb

BATCH_SIZE = 128
ATTENTION_THRESHOLD = 0.2
EVALUATION_DIR = './evaluation/ca-evaluation-3x3'
DATA_PATH = os.path.join(EVALUATION_DIR, 'trajectories_ep15000_dur36.20_ret0.52_5K.pt')
OBSERVATIONS_MEAN_PATH = './trajectories/ep15000_dur36.20_ret0.52/trajectories/10.00K_mean_s42.pt'  # computed on the train set
OBSERVATIONS_STD_PATH = './trajectories/ep15000_dur36.20_ret0.52/trajectories/10.00K_std_s42.pt'  # computed on the train set

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_evaluation():
    # 0. Connect wandb
    wandb.login()

    wandb.init(project='attentional_fw_baselines',
               entity='ut-rl-credit',
               tags=['SECRET', 'verification'],
               #  notes=note,
               config=dict(
                   attention_threshold=ATTENTION_THRESHOLD,
                   batch_size=BATCH_SIZE,
                   data_path=DATA_PATH,
               ))

    # 1. Load and pre-process data
    dataset = torch.load(DATA_PATH)
    dataset_size = dataset.observations.shape[0]

    # global variables used in other parts
    seq_len = dataset.observations.shape[1]
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_VAL)  # will be ignored
    timing = dict()  # will be ignored

    # credit assignment ground truths
    trigger_activations = find_activations(dataset.observations, dataset.actions, target='trigger').to(device)  # N, S
    prize_activations = find_activations(dataset.observations, dataset.actions, target='prize')  # N, S

    trigger_activations_indices = torch.argmax(trigger_activations, axis=-1).to(device)  # (N,) timesteps preceding trigger activation
    prize_activations_indices = torch.argmax(prize_activations, axis=-1)  # (N,) timesteps preceding taking prize

    episodes_with_prize_mask = torch.sum(prize_activations, axis=-1).to(device)  # mask for episodes where prizes were taken
    episodes_with_trigger_mask = torch.sum(trigger_activations, axis=-1).to(device)  # (N, )

    ca_gt = {
        'attention_threshold': ATTENTION_THRESHOLD,
        'batch_size': BATCH_SIZE,
        'episodes_with_trigger_mask': episodes_with_trigger_mask,
        'episodes_with_prize_mask': episodes_with_prize_mask,
        'seq_len': seq_len,
        'trigger_activations': trigger_activations,
        'trigger_timesteps': trigger_activations_indices,
        'prize_activations': prize_activations,
        'prize_timesteps': prize_activations_indices,
    }

    data_loader = preprocess_dataset(dataset, DATA_PATH, sum_rewards=True, normalize=True,
                                     eval_mode=True, path_to_mean=OBSERVATIONS_MEAN_PATH, path_to_std=OBSERVATIONS_STD_PATH)

    # 3. Evaluate each model on the data

    models = []
    for root, _, dirs in os.walk(EVALUATION_DIR):
        for path in dirs:
            if 'model' not in path:
                continue
            models.append(os.path.join(root, path))

    accs = []
    ca_precisions = []
    ca_recalls = []

    for model_path in models:
        model = torch.load(model_path).to(device)
        _, acc, ca_precision, ca_recall, rel_attention_vals = evaluate(model, criterion, data_loader, device, timing, ca_gt)

        accs.append(acc)
        ca_precisions.append(ca_precision)
        ca_recalls.append(ca_recall)

        fig, axes = plt.subplots(1, 1, figsize=(6, 6))
        x_axis = (torch.arange(rel_attention_vals.shape[0]) - seq_len).cpu()

        axes.set_title(f'Evaluation ({dataset_size} trajectories)')
        axes.set_xlabel('Relative step in trajectory (0 - activating trigger)')
        axes.set_ylabel('Average attention weights')
        axes.plot(x_axis, rel_attention_vals.cpu())

        wandb.log({
            'average_attention': wandb.Image(plt),
        })

    acc_mean, acc_std = torch.mean(torch.tensor(accs)), torch.std(torch.tensor(accs))
    precision_mean, precision_std = torch.mean(torch.tensor(ca_precisions)), torch.std(torch.tensor(ca_precisions))
    recall_mean, recall_std = torch.mean(torch.tensor(ca_recalls)), torch.std(torch.tensor(ca_recalls))

    # 5. Log the results table to Wandb

    data = [[f'{acc_mean:.3f} ± {acc_std:.3f}', f'{precision_mean:.3f} ± {precision_std:.3f}', f'{recall_mean:.3f} ± {recall_std:.3f}']]
    column_names = ["Accuracy (reward prediction)", "Precision (credit assignment)", "Recall (credit assignment)"]
    wandb.log({"results": wandb.Table(data=data, columns=column_names)})


if __name__ == '__main__':
    run_evaluation()

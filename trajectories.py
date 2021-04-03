import os
import math

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.functional import one_hot
from torch.nn.utils.rnn import pad_sequence


class TrajectoriesDataset(Dataset):
    def __init__(self, observations, actions, rewards):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return (self.observations[idx, :], self.actions[idx, :], self.rewards[idx, :], idx)


def preprocess_dataset(dataset, data_path, batch_size=128, valid_size=0.2, seed=42, sum_rewards=True, normalize=True):

    train_subset, valid_subset = validation_split(
        dataset, valid_size, seed=seed)

    if normalize:
        # calculate normalization values on train subset
        train_mean = torch.mean(
            dataset.observations[train_subset.indices], axis=(0, 1, 2, 3))
        train_std = torch.std(
            dataset.observations[train_subset.indices], axis=(0, 1, 2, 3))

        # save normalization values for future
        dataset_dirname = os.path.dirname(data_path)
        dataset_basename = '.'.join(
            os.path.basename(data_path).split('.')[:-1])
        torch.save(train_mean, os.path.join(dataset_dirname,
                                            f'{dataset_basename}_mean_s{seed}.pt'))
        torch.save(train_std, os.path.join(dataset_dirname,
                                           f'{dataset_basename}_std_s{seed}.pt'))

        # apply normalization on full dataset
        dataset.observations -= train_mean
        dataset.observations /= train_std

    # convert rewards to returns
    if sum_rewards:
        returns = torch.zeros_like(dataset.rewards)
        returns += torch.sum(dataset.rewards, axis=1)[:, None]
        dataset.rewards = returns

    # convert reward values (-1,0,1) into "classes" (0,1,2)
    dataset.rewards += 1

    # prepare batches
    train_loader = DataLoader(train_subset, batch_size=batch_size)
    valid_loader = DataLoader(valid_subset, batch_size=batch_size)

    return train_loader, valid_loader


def episodes_to_tensors(episodes, action_size, verbose=False, pad_val=10.):

    observations_by_episode = []
    actions_by_episode = []
    rewards_by_episode = []

    for episode in episodes:

        observations = []
        actions = []
        rewards = []

        for step in episode:
            observation, action, reward, done = step

            observations.append(observation)
            actions.append(one_hot_encode_action(action, action_size))
            rewards.append(reward)

        observations = torch.tensor(observations, dtype=torch.float32)
        actions = torch.stack(actions).type(torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.long)

        observations_by_episode.append(observations)
        actions_by_episode.append(actions)
        rewards_by_episode.append(rewards)

    observations_by_episode = pad_sequence(
        observations_by_episode, batch_first=True, padding_value=0)
    actions_by_episode = pad_sequence(
        actions_by_episode, batch_first=True, padding_value=pad_val)
    rewards_by_episode = pad_sequence(
        rewards_by_episode, batch_first=True, padding_value=0)  # pad reward with 0 because we only case about return anyways

    return observations_by_episode, actions_by_episode, rewards_by_episode


def one_hot_encode_action(action, n_classes=3):
    index = torch.tensor(int(action))
    return one_hot(index, n_classes)


def save_trajectories(trajectories, base_dir, agent_dir, action_size, pad_val=10):
    print('Saving experience...')
    # convert list of episodes into PyTorch dataset
    observations, actions, rewards = episodes_to_tensors(
        trajectories, action_size, verbose=False, pad_val=pad_val)
    dataset = TrajectoriesDataset(observations, actions, rewards)

    # create a new folder for dataset
    directory = os.path.join(base_dir, agent_dir, 'trajectories')
    os.makedirs(directory, exist_ok=True)
    recording_name = f'{len(trajectories)/1000:.2f}K.pt'
    full_path = os.path.join(directory, recording_name)

    torch.save(dataset, full_path)


def find_activations(observations, actions, forward_action=2, target='prize'):
    '''
    Find steps at which the agent activated a target object by stepping onto it.

    observations - tensor of shape (N, S, H, W, C)
    actions - tensor of shape (N, S, A)
    target – one of {trigger|prize}

    return: binary mask (N, S)
    '''

    target_color = None
    if target == 'prize':
        target_color = [255, 76, 249]  # pink
    if target == 'trigger':
        target_color = [255, 76, 76]  # red

    N, S, H, W, _ = observations.shape
    center_h = H // 2
    center_w = W // 2
    out_shape = (N, S)

    centers = observations[:, :, center_h, center_w, :]  # N, S, C
    action_scalars = torch.argmax(actions, axis=-1)  # N, S
    trigger_ahead = (centers == torch.tensor(target_color)).all(axis=-1)
    going_forward = (action_scalars == forward_action)

    return torch.logical_and(trigger_ahead, going_forward, out=torch.empty(out_shape, dtype=torch.uint8))


def normalize_observations(dataset):
    mean = torch.mean(dataset.observations, axis=(0, 1, 2, 3))
    std = torch.std(dataset.observations, axis=(0, 1, 2, 3))

    dataset.observations -= mean
    dataset.observations /= std

    return mean, std


def validation_split(dataset, validation_subset, seed=42):

    if validation_subset > 0:
        n_total_samples = len(dataset)
        n_train_samples = math.floor(n_total_samples * (1-validation_subset))
        n_valid_samples = n_total_samples - n_train_samples

        train_subset, valid_subset = random_split(
            dataset,
            [n_train_samples, n_valid_samples],
            generator=torch.Generator().manual_seed(seed)
        )  # reproducible results

    else:
        train_subset = dataset
        valid_subset = None

    return train_subset, valid_subset

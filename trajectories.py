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
        return (self.observations[idx, :], self.actions[idx, :], self.rewards[idx, :])


def load_trajectories(path_to_dataset):
    dataset = torch.load(path_to_dataset)

    return dataset


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


def normalize_observations(dataset):
    mean = torch.mean(dataset.observations, axis=(0, 1, 2, 3))
    std = torch.std(dataset.observations, axis=(0, 1, 2, 3))

    dataset.observations -= mean
    dataset.observations /= std

    return mean, std


# TODO: choose validation indices first, and then use them
# to count normalization values (mean, std) only on the train subset
def get_data_loaders(dataset, batch_size=1024, validation_subset=0, seed=42, verbose=False):

    mean, std = normalize_observations(dataset)

    if validation_subset > 0:
        n_total_samples = len(dataset)
        n_train_samples = math.floor(n_total_samples * (1-validation_subset))
        n_valid_samples = n_total_samples - n_train_samples

        train_dataset, valid_dataset = random_split(
            dataset,
            [n_train_samples, n_valid_samples],
            generator=torch.Generator().manual_seed(seed)
        )  # reproducible results

        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

        if verbose:
            print('Train set size:', len(train_dataset), 'samples')
            print('Train set:', len(train_loader), 'batches')
            print('Validation set size:', len(valid_dataset), 'samples')
            print('Validation set:', len(valid_loader), 'batches')
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size)
        valid_loader = None
        if verbose:
            print('Prepared:', len(train_loader), 'batches')

    return train_loader, valid_loader

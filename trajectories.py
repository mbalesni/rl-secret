import os
import math

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.functional import one_hot

from config import PAD_VAL


class TrajectoriesDataset(Dataset):
    def __init__(self, observations, actions, rewards):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return (self.observations[idx, :], self.actions[idx, :], self.rewards[idx, :], idx)

    def compute_pad_mask(self):
        return self.actions[:, :, 0] != PAD_VAL


class StateFullTrajectoriesDataset(Dataset):
    def __init__(self, observations, full_observations, actions, rewards):
        self.observations = observations
        self.full_observations = full_observations
        self.actions = actions
        self.rewards = rewards

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return (self.observations[idx, :], self.full_observations[idx, :], self.actions[idx, :], self.rewards[idx, :], idx)

    def compute_pad_mask(self):
        return self.actions[:, :, 0] != PAD_VAL


class TrajectoriesRecorder:
    def __init__(self, n_episodes, seq_len, obs_dims, act_size, save_path, pad_val=PAD_VAL):

        assert save_path is not None
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        self.observations = torch.zeros((n_episodes, seq_len, *obs_dims), dtype=torch.float32)
        self.actions = torch.ones((n_episodes, seq_len, act_size), dtype=torch.float32) * pad_val
        self.rewards = torch.ones((n_episodes, seq_len), dtype=torch.long) * int(pad_val)

        self.act_size = act_size
        self.save_path = save_path

        self.current_episode = 0
        self.current_step = 0

    def add_step(self, observation, action, reward, done):

        self.observations[self.current_episode][self.current_step] = torch.tensor(observation, dtype=torch.float32)

        # one-hot encode actions
        self.actions[self.current_episode][self.current_step] = one_hot_encode_action(action, self.act_size).type(torch.float32)

        # convert reward values (-1,0,1) into "classes" (0,1,2)
        self.rewards[self.current_episode][self.current_step] = reward + 1

        self.current_step += 1

        if done:
            self.current_episode += 1
            self.current_step = 0

    def save(self):
        dataset = TrajectoriesDataset(self.observations, self.actions, self.rewards)
        torch.save(dataset, self.save_path)


class StatefullTrajectoriesRecorder:
    def __init__(self, n_episodes, seq_len, obs_dims, full_obs_dims, act_size, save_path, pad_val=PAD_VAL):

        assert save_path is not None
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        self.observations = torch.zeros((n_episodes, seq_len, *obs_dims), dtype=torch.float32)
        self.full_observations = torch.zeros((n_episodes, seq_len, *full_obs_dims), dtype=torch.float32)
        self.actions = torch.ones((n_episodes, seq_len, act_size), dtype=torch.float32) * pad_val
        self.rewards = torch.ones((n_episodes, seq_len), dtype=torch.long) * int(pad_val)

        self.act_size = act_size
        self.save_path = save_path

        self.current_episode = 0
        self.current_step = 0

    def add_step(self, observation, full_observation, action, reward, done):

        # add 1) partial (cropped) pixel observation and 2) full pixel observation
        self.observations[self.current_episode][self.current_step] = torch.tensor(observation, dtype=torch.float32)
        self.full_observations[self.current_episode][self.current_step] = torch.tensor(full_observation, dtype=torch.float32)

        # one-hot encode actions
        self.actions[self.current_episode][self.current_step] = one_hot_encode_action(action, self.act_size).type(torch.float32)

        # convert reward values (-1,0,1) into "classes" (0,1,2)
        self.rewards[self.current_episode][self.current_step] = reward + 1

        self.current_step += 1

        if done:
            self.current_episode += 1
            self.current_step = 0

    def save(self):
        dataset = StateFullTrajectoriesDataset(self.observations, self.full_observations, self.actions, self.rewards)
        torch.save(dataset, self.save_path)


def preprocess_dataset(dataset, data_path, batch_size=128, valid_size=0.2, seed=42,
                       sum_rewards=True, eval_mode=False, path_to_mean=None, path_to_std=None, normalize=True):

    observations_mean_path = None
    observations_std_path = None

    if not eval_mode:
        train_subset, valid_subset = validation_split(
            dataset, valid_size, seed=seed)

    if normalize:

        if not eval_mode:
            # calculate normalization values on train subset
            train_mean = torch.mean(
                dataset.observations[train_subset.indices], axis=(0, 1, 2, 3))
            train_std = torch.std(
                dataset.observations[train_subset.indices], axis=(0, 1, 2, 3))

            # save normalization values for future
            dataset_dirname = os.path.dirname(data_path)
            dataset_basename = '.'.join(
                os.path.basename(data_path).split('.')[:-1])

            observations_mean_path = os.path.join(dataset_dirname, f'{dataset_basename}_mean.pt')
            observations_std_path = os.path.join(dataset_dirname, f'{dataset_basename}_std.pt')

            torch.save(train_mean, observations_mean_path)
            torch.save(train_std, observations_std_path)
        else:
            train_mean = torch.load(path_to_mean)
            train_std = torch.load(path_to_std)

        # apply normalization on full dataset
        dataset.observations -= train_mean
        dataset.observations /= train_std

    # convert rewards to returns
    if sum_rewards:
        padding_mask = (dataset.actions[:, :, 0] == PAD_VAL)  # PAD_VAL are True
        dataset.rewards -= 1  # turn reward classes [0,1,2] into rewards [-1,0,1]
        dataset.rewards[padding_mask] *= 0

        returns = torch.zeros_like(dataset.rewards)
        returns += torch.sum(dataset.rewards, axis=1)[:, None]
        returns += 1  # turn returns [-1,0,1] into return classes [0,1,2]
        returns[padding_mask] = PAD_VAL

        dataset.rewards = returns

    # prepare batches
    if not eval_mode:
        train_loader = DataLoader(train_subset, batch_size=batch_size)
        valid_loader = DataLoader(valid_subset, batch_size=batch_size)

        return train_loader, valid_loader, observations_mean_path, observations_std_path
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader


def one_hot_encode_action(action, n_classes=3):
    index = torch.tensor(int(action))
    return one_hot(index, n_classes)


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

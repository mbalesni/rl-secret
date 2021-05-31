import torch
import os
from tqdm import tqdm

device = torch.device('cuda')


class SECRETRewardShaper:
    def __init__(self, model_path, shaping_path, dataset_path=None, use_full_obs=True):
        assert os.path.exists(model_path), 'Must pass the path to an existing reward prediction model'
        assert os.path.exists(shaping_path) or os.path.exists(dataset_path), 'Must pass either existing shaping or dataset'
        os.makedirs(os.path.dirname(shaping_path), exist_ok=True)

        self.use_full_obs = use_full_obs

        self.reward_predictor = torch.load(model_path)

        if dataset_path is None:
            self.potential = torch.load(shaping_path)
        else:
            self.compute_potential(dataset_path)

    def compute_potential(self, dataset_path):
        dataset = torch.load(dataset_path)
        dataset.rewards -= 1  # turn reward states to actual rewards
        pad_mask = dataset.compute_pad_mask()
        dataset.rewards *= pad_mask

        # if self.use_full_obs:
        #     flat_states = torch.flatten(dataset.full_observations, start_dim=0, end_dim=1)
        #     print('using full obs')
        # else:
        #     flat_states = torch.flatten(dataset.observations, start_dim=0, end_dim=1)
        #     print('using partial obs')

        # self.idx_to_state = torch.unique(flat_states, dim=0)
        # print('States shape before:', dataset.full_observations.shape)
        # print('Unique states shape:', self.idx_to_state.shape)
        # self.state_to_idx = {state: idx for idx, state in enumerate(list(self.idx_to_state))}
        # self.potential = {state: 0 for state in self.state_to_idx.keys()}
        self.potential = {}

        n_trajectories = len(dataset.observations)

        # for unique_state in tqdm(self.potential.keys(), mininterval=0.5):
        # unique_state = dataset.full_observations[0, 0]

        pbar = tqdm(range(n_trajectories))

        for i_trajectory in pbar:
            trajectory = (dataset.observations[i_trajectory], dataset.full_observations[i_trajectory],
                          dataset.actions[i_trajectory], dataset.rewards[i_trajectory])
            attention = self.get_trajectory_attention(trajectory)

            for i_step in range(len(trajectory[0])):
                step_state = trajectory[1][i_step]

                # if not (step_state == unique_state).all():
                #     continue

                if i_step == 0:  # protect against negative indexes
                    continue

                query_state = trajectory[1][i_step - 1]
                query_action = trajectory[2][i_step - 1]

                state_indx = tuple(step_state.flatten().tolist())
                if state_indx not in self.potential:
                    self.potential[state_indx] = 0
                self.potential[state_indx] += self.redistributed_return(trajectory, query_state, query_action, attention)

            pbar.set_postfix_str(f'Uniques states: {len(self.potential.keys())}')

        for unique_state in self.potential.keys():
            self.potential[unique_state] /= n_trajectories

    def get_trajectory_attention(self, trajectory):
        observations, full_observations, actions, rewards = trajectory

        observations_cuda = observations.unsqueeze(1).transpose(2, 4).to(device)
        actions_cuda = actions.unsqueeze(1).to(device)
        # rewards_cuda = rewards.unsqueeze(1).to(device)

        # TODO: only use attention from correctly predicted timesteps
        _, attention_output_cuda = self.reward_predictor(observations_cuda, actions_cuda, output_attention=True)  # attention: (N, L, S)

        return attention_output_cuda

    def redistributed_return(self, trajectory, query_state, query_action, attention):
        observations, full_observations, actions, rewards = trajectory
        return_sum = 0

        rewards_cuda = rewards.unsqueeze(1).to(device)

        for i_step in range(len(observations)):

            if not ((full_observations[i_step] == query_state).all() and (actions[i_step] == query_action).all()):
                continue

            return_sum += torch.dot(attention[0, :, i_step].squeeze(),
                                    rewards_cuda.type(torch.float).squeeze())  # attention at all steps towards `i_step`

        return return_sum

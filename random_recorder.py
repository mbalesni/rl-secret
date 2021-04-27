import numpy as np
import gym
import torch
import click
import os
import random
from xvfbwrapper import Xvfb

import wandb
from itertools import count

from gym_minigrid.wrappers import RGBImgObsWrapper
from tqdm import tqdm
from trajectories import TrajectoriesRecorder, StatefullTrajectoriesRecorder


def frames_to_video(frames, fps=15):
    stacked_frames = np.array(frames).transpose(
        (0, 3, 1, 2))  # (h,w,c) -> (t,c,w,h)
    return wandb.Video(stacked_frames, fps=fps, format="gif")


def record(episodes, env, part_obs_shape, full_obs_shape, obs_shape,
           part_obs_key, full_obs_key, obs_key, save_path, log_frequency,
           both_observ=False, log_prefix='', gifs=False):
    '''Record and save a dataset of trajectories by a random policy.'''

    if both_observ:
        trajectories_recorder = StatefullTrajectoriesRecorder(episodes, env.max_steps, part_obs_shape, full_obs_shape, env.action_space.n, save_path)
    else:
        trajectories_recorder = TrajectoriesRecorder(episodes, env.max_steps, obs_shape, env.action_space.n, save_path)

    frames = []
    logs = {
        'episode_durations': [],
        'episode_returns': []
    }
    timing = dict()
    for i_episode in tqdm(range(1, episodes+1), mininterval=0.5):

        observation = env.reset()
        if gifs:
            frames = [env.render(mode='rgb_array')]
        rewards = []

        for i_step in count():
            partial_observation = observation[part_obs_key]
            full_observation = observation[full_obs_key]
            agent_observation = observation[obs_key]

            action = np.random.choice(env.action_space.n, p=[0.2, 0.2, 0.6])  # [left, right, forward] go forward more often
            observation, reward, done, _ = env.step(action)

            if gifs:
                frames.append(env.render(mode='rgb_array'))
            rewards.append(reward)

            if both_observ:
                trajectories_recorder.add_step(partial_observation, full_observation, action, reward, done)
            else:
                trajectories_recorder.add_step(agent_observation, action, reward, done)

            if done:
                logs['episode_returns'].append(sum(rewards))
                logs['episode_durations'].append(len(rewards))
                break

        if i_episode > 1 and (i_episode % log_frequency) == 0:

            avg_duration = np.mean(logs['episode_durations'])
            avg_return = torch.mean(torch.tensor(logs['episode_returns'], device='cpu', dtype=torch.float))

            wandb.log({
                f'{log_prefix}avg_episode_duration': avg_duration,
                f'{log_prefix}avg_episode_return': avg_return,
                f'{log_prefix}video': frames_to_video(frames) if gifs else None,
                f'{log_prefix}episode': i_episode,
                **{f'{log_prefix}{k}': v['time'] / v['count'] for k, v in timing.items()}
            })
            logs['episode_durations'] = []
            logs['episode_returns'] = []
            timing = dict()

    print('Saving...', end=' ')
    trajectories_recorder.save()
    print('Done!')


@click.command()
@click.option('--note', type=str, help='message to explain how is this run different')
@click.option('--env-name', type=str, required=True)  # e.g. MiniGrid-Triggers-3x3-T1P1-v0
@click.option('--env-label', type=str, required=True, help='environment label used in path')  # e.g. MiniGrid-Triggers-3x3-T1P1-v0
@click.option('--episodes', type=int, help='number of episodes to record for', required=True)
@click.option('--gifs/--no-gifs', help='log gifs of some episodes to wandb', default=False)
@click.option('--part-observ/--full-observ', required=True, default=None, help='view a part of the environment or see the full picture')
@click.option('--both-observ/--single-observ', required=False, default=False, help='whether to save both partial + full observations')
@click.option('--seed', type=int, default=42, help='random seed used')
@click.option('--log-frequency', type=int, default=5e1, help='logging frequency, episodes')
@click.option('--test-episodes', type=int, help='number of test episodes to record for', required=True)
@click.option('--use-wandb/--no-wandb', default=True)
def run(note, env_name, env_label, episodes, gifs, part_observ, both_observ, seed, log_frequency, test_episodes, use_wandb):
    global trajectories
    global test_trajectories

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = gym.make(env_name)
    env = RGBImgObsWrapper(env)

    observation = env.reset()
    part_obs_key = 'image_partial'
    full_obs_key = 'image'
    obs_key = part_obs_key if part_observ else full_obs_key
    part_obs_shape = observation[part_obs_key].shape
    full_obs_shape = observation[full_obs_key].shape
    obs_shape = observation[obs_key].shape

    agent_dir = f'agents/{env_label}/random'
    trajectories_path = os.path.join(agent_dir, 'trajectories', f'train_{episodes}.pt')
    test_trajectories_path = os.path.join(agent_dir, 'trajectories', f'test_{test_episodes}.pt')

    wandb.login()
    wandb.init(entity='ut-rl-credit',
               project='attentional_fw_baselines',
               tags=['SECRET', 'verification'],
               notes=note or 'Collect trajectories with random policy ',
               mode='online' if use_wandb else 'disabled',
               config=dict(
                    both_observ=both_observ,
                    env_name=env_name,
                    env_label=env_label,
                    episodes=episodes,
                    obs_shape=obs_shape,
                    log_frequency=log_frequency,
                    part_observ=part_observ,
                    seed=seed,
                    test_episodes=test_episodes,
                    output_trajectories_path=trajectories_path,
                    output_test_trajectories_path=test_trajectories_path,
               ),
               reinit=True)

    with Xvfb():
        print('\nRecording training trajectories...\n')
        record(episodes, env, part_obs_shape, full_obs_shape, obs_shape, part_obs_key, full_obs_key, obs_key, trajectories_path,
               log_frequency, both_observ=both_observ, gifs=gifs)

        print('\nRecording test trajectories...\n')
        record(test_episodes, env, part_obs_shape, full_obs_shape, obs_shape, part_obs_key, full_obs_key, obs_key, test_trajectories_path,
               log_frequency, both_observ=both_observ, log_prefix='test_', gifs=gifs)

    wandb.finish()
    env.close()


if __name__ == '__main__':
    run()

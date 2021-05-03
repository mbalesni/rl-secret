import numpy as np
import gym
import torch
import click
import os
import random
from xvfbwrapper import Xvfb

import wandb
from itertools import count
from timing import Timing
from tqdm import tqdm

from gym_minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
from dqn_pfrl import DQN_PFRL_ACTOR


def frames_to_video(frames, fps=15):
    stacked_frames = np.array(frames).transpose((0, 3, 1, 2))  # (h,w,c) -> (t,c,w,h)
    return wandb.Video(stacked_frames, fps=fps, format="gif")


def compute_obs_normalization(episodes, env, obs_shape, save_dir):

    observations = torch.zeros((episodes, 50, *obs_shape))
    with Xvfb():
        for i_episode in tqdm(range(episodes), mininterval=0.5, desc='Computing obs norm'):

            observation = env.reset()

            for i_step in count():
                action = np.random.choice(env.action_space.n, p=[0.2, 0.2, 0.6])  # [left, right, forward] go forward more often
                observation, reward, done, _ = env.step(action)
                observations[i_episode, i_step] = torch.tensor(observation)

                if done:
                    break

    # calculate normalization values on train subset
    mean = torch.mean(observations, axis=(0, 1, 2, 3))
    std = torch.std(observations, axis=(0, 1, 2, 3))

    # save normalization values for potential future use
    os.makedirs(save_dir, exist_ok=True)
    torch.save(mean, os.path.join(save_dir, 'obs_mean.pt'))
    torch.save(std, os.path.join(save_dir, 'obs_std.pt'))

    print('mean', mean)
    print('std', std)

    return mean, std


def run_training(episodes, agent, env, obs_shape, save_path, log_frequency, log_prefix='', gifs=False, normalize=False):
    '''Run training of a DQN PFRL agent with SECRET attention model and custom reward redistribution.'''

    frames = []
    logs = {
        'episode_durations': [],
        'episode_returns': []
    }
    timing = dict()

    best_return = -999999
    best_agent_dir = os.path.join(save_path, 'best_agent')
    final_agent_dir = os.path.join(save_path, 'final_agent')
    os.makedirs(best_agent_dir, exist_ok=True)
    os.makedirs(final_agent_dir, exist_ok=True)

    if normalize:
        obs_mean, obs_std = compute_obs_normalization(100, env, obs_shape, save_path)
        obs_mean_np = obs_mean.numpy()
        obs_std_np = obs_std.numpy()

    actor = agent.agent

    with Xvfb():
        for i_episode in tqdm(range(1, episodes+1), mininterval=0.5):

            observation = env.reset()
            if gifs:
                frames = [env.render(mode='rgb_array')]
            rewards = []

            for i_step in count():

                with Timing(timing, 'time_choose_act'):
                    action = actor.act(observation)

                with Timing(timing, 'time_perform_act'):
                    observation, reward, done, _ = env.step(action)

                if normalize:
                    with Timing(timing, 'time_normalize_obs'):
                        observation = observation.astype(np.float32)
                        observation -= obs_mean_np
                        observation /= obs_std_np

                with Timing(timing, 'time_observe'):
                    actor.observe(observation, reward, done, reset=False)

                if gifs:
                    frames.append(env.render(mode='rgb_array'))
                rewards.append(reward)

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

                if avg_return > best_return:
                    best_return = avg_return
                    actor.save(best_agent_dir)


@click.command()
@click.option('--batch-size', default=32, help='training batch size')
@click.option('--buffer-size', default=5e4, type=int, help='size of the replay buffer')
@click.option('--buffer-prefill-steps', default=5e3, type=int, help='size of the prefilled replay buffer')
@click.option('--env-name', type=str, default='MiniGrid-Triggers-3x3-T1P1-v0')
@click.option('--episodes', type=int, help='number of episodes to train for', required=True)
@click.option('--eps-start', type=float, default=1, help='starting exploration epsilon')
@click.option('--eps-end', type=float, default=0.1, help='final exploration epsilon')
@click.option('--eps-decay', type=int, default=250_000, help='final exploration epsilon')
@click.option('--gamma', type=float, default=0.999, help='discount factor')
@click.option('--group', type=str)
@click.option('--log-frequency', type=int, default=5e1, help='logging frequency, episodes')
@click.option('--learning-rate', type=float, default=1e-1, help='goal learning rate')
@click.option('--note', type=str, help='message to explain how is this run different')
@click.option('--normalize-obs/--not-normalize-obs', default=None, required=True, help='whether to normalize agent observations')
@click.option('--seed', type=int, default=42, help='random seed used')
@click.option('--target-freq', type=int, default=2000, help='how frequently to update target network')
@click.option('--update-freq', type=int, default=1, help='how frequently to run optimization')
@click.option('--use-wandb/--no-wandb', default=True)
def train(batch_size, buffer_size, buffer_prefill_steps,
          env_name, episodes, eps_start, eps_end, eps_decay,
          gamma, group,
          log_frequency, learning_rate,
          note, normalize_obs,
          seed,
          target_freq,
          update_freq, use_wandb):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    wandb.login()
    wandb.init(entity='ut-rl-credit',
               project='attentional_fw_baselines',
               group=group,
               notes=note,
               mode='online' if use_wandb else 'disabled',
               config=dict(
                   batch_size=batch_size,
                   buffer_prefill_steps=buffer_prefill_steps,
                   buffer_size=buffer_size,
                   env_name=env_name,
                   episodes=episodes,
                   eps_decay=eps_decay,
                   eps_end=eps_end,
                   eps_start=eps_start,
                   gamma=gamma,
                   learning_rate=learning_rate,
                   log_frequency=log_frequency,
                   normalize_obs=normalize_obs,
                   seed=seed,
                   target_freq=target_freq,
                   update_freq=update_freq,
               ))
    # Upload models at the end of training
    save_dir = wandb.run.dir if use_wandb else './'
    agent_save_dir = os.path.join(save_dir, 'agents', env_name)

    os.makedirs(agent_save_dir, exist_ok=True)
    wandb.save(os.path.join(save_dir, "*.pt"))
    wandb.save(os.path.join(agent_save_dir, "*.pt"))

    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = gym.make(env_name)
    if 'MiniGrid' in env_name:  # support non-MiniGrid environments
        env = ImgObsWrapper(RGBImgObsWrapper(env))

    observation_shape = env.reset().shape
    agent = DQN_PFRL_ACTOR(*observation_shape, env.action_space.n,
                           learning_rate=learning_rate, buffer_size=buffer_size,
                           gamma=gamma, batch_size=batch_size, eps_start=eps_start,
                           eps_end=eps_end, eps_decay=eps_decay, update_interval=update_freq,
                           target_update=target_freq, env=env, update_start=buffer_prefill_steps)

    run_training(episodes, agent, env, observation_shape, agent_save_dir, log_frequency, normalize=normalize_obs)

    env.close()
    wandb.finish()


if __name__ == '__main__':
    train()

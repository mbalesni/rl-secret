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

from gym_minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper, RGBImgObsWrapper
from dqn_pfrl import DQN_PFRL_ACTOR

logs = {
    'episode_durations': [],
    'episode_returns': [],
    'rec_episode_durations': [],
    'rec_episode_returns': [],
}


def frames_to_video(frames, fps=15):
    '''According to: https://docs.wandb.ai/library/log#images-and-overlays'''
    stacked_frames = np.array(frames).transpose(
        (0, 3, 1, 2))  # (h,w,c) -> (t,c,w,h)
    return wandb.Video(stacked_frames, fps=fps, format="gif")


def record_experience(actor, env, po_env, n_episodes, log_frequency):
    timing = dict()
    recording = {
        'episodes': []
    }
    print('Recording...')
    with actor.eval_mode():
        for i_episode in range(1, n_episodes+1):
            observation = env.reset()
            recording['episodes'].append([])
            frames = [env.render(mode='rgb_array')]
            rewards = []

            for i_step in count():
                with Timing(timing, 'time_choose_act'):
                    action = actor.act(observation)

                with Timing(timing, 'time_perform_act'):
                    observation, reward, done, _ = env.step(action)

                with Timing(timing, 'time_observe'):
                    actor.observe(observation, reward, done, reset=False)

                frames.append(env.render(mode='rgb_array'))
                rewards.append(reward)

                partial_observation = po_env.render(mode='rgb_array')
                step = (partial_observation, action, reward, done)
                recording['episodes'][i_episode].append(step)

                if done:
                    logs['rec_episode_returns'].append(sum(rewards))
                    logs['rec_episode_durations'].append(len(rewards))
                    break

            if i_episode > 1 and (i_episode % log_frequency) == 0:

                avg_duration = np.mean(logs['rec_episode_durations'])
                avg_return = torch.mean(torch.tensor(
                    logs['rec_episode_returns'], device='cpu', dtype=torch.float))

                wandb.log({
                    'rec_avg_episode_duration': avg_duration,
                    'rec_avg_episode_return': avg_return,
                    'rec_video': frames_to_video(frames),
                    'rec_episode': i_episode,
                    'rec_agent_step': actor.cumulative_steps,
                    'rec_epsilon': actor.explorer.epsilon,
                    **{f'rec_{k}': v['time'] / v['count'] for k, v in timing.items()}
                }, )
                logs['episode_durations'] = []
                logs['episode_returns'] = []
                timing = dict()

    return recording


@click.command()
@click.option('--batch-size', default=32, help='training batch size')
@click.option('--buffer-size', default=5e4, type=int, help='size of the replay buffer')
@click.option('--buffer-prefill-steps', default=5e3, type=int, help='size of the prefilled replay buffer')
@click.option('--checkpoint-frequency', type=int, default=5e3, help='experience recording frequency, episodes')
@click.option('--checkpoint-episodes', type=int, default=1e4, help='number of episodes to record at each checkpoint')
@click.option('--checkpoint-models', type=int, default=1e4, help='number of models to train on each checkpoint\'s agent\'s collected trajectories')
@click.option('--env-name', type=str, default='MiniGrid-Triggers-3x3-v0', required=True)
@click.option('--episodes', type=int, help='number of episodes to train for', required=True)
@click.option('--eps-start', type=float, default=1, help='starting exploration epsilon')
@click.option('--eps-end', type=float, default=0.1, help='final exploration epsilon')
@click.option('--eps-decay', type=int, default=250_000, help='over how many steps to decay epsilon')
@click.option('--gamma', type=float, default=0.999, help='discount factor')
@click.option('--group', type=str, help='common name to group the end-to-end experiment')
@click.option('--learning-rate', type=float, default=1e-1, help='goal learning rate')
@click.option('--log-frequency', type=int, default=5e1, help='logging frequency, episodes')
@click.option('--note', type=str, help='message to explain how is this run different')
@click.option('--target-freq', type=int, default=2000, help='how frequently to update target network')
@click.option('--seed', type=int, default=42, help='random seed used')
@click.option('--update-freq', type=int, default=1, help='how frequently to run optimization')
@click.option('--use-wandb/--no-wandb', default=True)
def train(batch_size, buffer_size, buffer_prefill_steps,
          checkpoint_frequency, checkpoint_episodes, checkpoint_models,
          env_name, episodes, eps_start, eps_end, eps_decay,
          gamma, group,
          learning_rate, log_frequency,
          note,
          target_freq,
          seed,
          update_freq,
          use_wandb):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    wandb.login()
    wandb.init(entity='ut-rl-credit',
               project='attentional_fw_baselines',
               group=group,
               notes=note,
               tags=["SECRET"],
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
                   rec_episodes=checkpoint_episodes,
                   rec_frequency=checkpoint_frequency,
                   seed=seed,
                   target_freq=target_freq,
                   update_freq=update_freq,
               ))

    # Upload models at the end of training
    save_dir = wandb.run.dir if use_wandb else './'

    wandb.save(os.path.join(save_dir, "*.pt"))
    wandb.save(os.path.join(save_dir, "recordings", "*.pt"))
    os.makedirs(os.path.join(save_dir, 'agent'), exist_ok=True)

    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = gym.make(env_name)
    if 'MiniGrid' in env_name:  # support non-MiniGrid environments
        partial_obs_env = ImgObsWrapper(RGBImgPartialObsWrapper(env))
        if part_observ:
            env = ImgObsWrapper(RGBImgPartialObsWrapper(env))
        else:
            env = ImgObsWrapper(RGBImgObsWrapper(env))

    observation_shape = env.reset().shape
    actor = DQN_PFRL_ACTOR(*observation_shape, env.action_space.n,
                           learning_rate=learning_rate, buffer_size=buffer_size,
                           gamma=gamma, batch_size=batch_size, eps_start=eps_start,
                           eps_end=eps_end, eps_decay=eps_decay, update_interval=update_freq,
                           target_update=target_freq, env=env, update_start=buffer_prefill_steps).agent

    with Xvfb():

        timing = dict()
        print('Training...')
        for i_episode in range(1, episodes+1):
            observation = env.reset()
            frames = [env.render(mode='rgb_array')]
            rewards = []

            for i_step in count():
                with Timing(timing, 'time_choose_act'):
                    action = actor.act(observation)

                with Timing(timing, 'time_perform_act'):
                    observation, reward, done, _ = env.step(action)

                with Timing(timing, 'time_observe'):
                    actor.observe(observation, reward, done, reset=False)

                frames.append(env.render(mode='rgb_array'))
                rewards.append(reward)

                if done:
                    logs['episode_returns'].append(sum(rewards))
                    logs['episode_durations'].append(len(rewards))
                    break

            if i_episode > 1 and (i_episode % log_frequency) == 0:

                avg_duration = np.mean(logs['episode_durations'])
                avg_return = torch.mean(torch.tensor(
                    logs['episode_returns'], device='cpu', dtype=torch.float))

                wandb.log({
                    'avg_episode_duration': avg_duration,
                    'avg_episode_return': avg_return,
                    'video': frames_to_video(frames),
                    'episode': i_episode,
                    'agent_step': actor.cumulative_steps,
                    'epsilon': actor.explorer.epsilon,
                    **{k: v['time'] / v['count'] for k, v in timing.items()}
                }, )
                logs['episode_durations'] = []
                logs['episode_returns'] = []
                timing = dict()

                if record and i_episode % rec_frequency == 0:
                    recording = record_experience(actor, env, partial_obs_env, rec_episodes, log_frequency)
                    agent_version_folder = f'ep{i_episode}_dur{avg_duration:.2f}_ret{avg_return:.2f}'
                    path_to_recording = os.path.join(save_dir, 'recordings', agent_version_folder)
                    save_experience(recording, path_to_recording)

    env.close()
    wandb.finish()


if __name__ == '__main__':
    train()

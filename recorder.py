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

from gym_minigrid.wrappers import RGBImgObsWrapper
from dqn_pfrl import DQN_PFRL_ACTOR
from tqdm import tqdm
from trajectories import save_trajectories

trajectories = []

logs = {
    'episode_durations': [],
    'episode_returns': []
}


def frames_to_video(frames, fps=15):
    stacked_frames = np.array(frames).transpose(
        (0, 3, 1, 2))  # (h,w,c) -> (t,c,w,h)
    return wandb.Video(stacked_frames, fps=fps, format="gif")


@click.command()
@click.option('--note', type=str, help='message to explain how is this run different')
@click.option('--env-name', type=str, default='MiniGrid-Triggers-3x3-T1P1-v0')
@click.option('--seed', type=int, default=42, help='random seed used')
@click.option('--log-frequency', type=int, default=5e1, help='logging frequency, episodes')
@click.option('--episodes', type=int, help='number of episodes to record for', required=True)
@click.option('--model-dir', type=int, help='path to agent models', required=False)
@click.option('--use-wandb/--no-wandb', default=True)
def record(note, env_name, seed, log_frequency, episodes, model_dir, use_wandb):
    global trajectories

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    wandb.login()

    save_dir = './'

    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = gym.make(env_name)
    if 'MiniGrid' in env_name:  # support non-MiniGrid environments
        env = RGBImgObsWrapper(env)

    observation_shape = env.reset()['image'].shape

    trajectories_base_dir = os.path.join(save_dir, './trajectories')

    agent_dirs = []
    for root, dirs, files in os.walk(trajectories_base_dir):
        agent_dirs = dirs
        break

    for agent_dir in agent_dirs:
        wandb.init(project='attentional_fw_baselines',
                   entity='ut-rl-credit',
                   notes='Collecting trajectories with one of the RL agents',
                   mode='online' if use_wandb else 'disabled',
                   config=dict(
                       agent=agent_dir,
                       env_name=env_name,
                       seed=seed,
                       log_frequency=log_frequency,
                       episodes=episodes,
                   ),
                   reinit=True)

        actor = DQN_PFRL_ACTOR(*observation_shape, 3, env=env).agent
        actor.load(os.path.join(trajectories_base_dir, agent_dir, 'agent'))

        with Xvfb():
            with actor.eval_mode():

                timing = dict()
                print('Recording...')
                for i_episode in tqdm(range(1, episodes+1)):
                    observation = env.reset()
                    if record:
                        trajectories.append([])
                    frames = [env.render(mode='rgb_array')]
                    rewards = []

                    for i_step in count():
                        partial_observation = observation['image_partial']

                        with Timing(timing, 'time_choose_act'):
                            action = actor.act(observation['image'])

                        with Timing(timing, 'time_perform_act'):
                            observation, reward, done, _ = env.step(action)

                        with Timing(timing, 'time_observe'):
                            actor.observe(
                                observation['image'], reward, done, reset=False)

                        frames.append(env.render(mode='rgb_array'))
                        rewards.append(reward)

                        # record cropped, *partial* observation for reward-prediction model
                        step = (partial_observation, action, reward, done)
                        trajectories[i_episode-1].append(step)

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

        save_trajectories(trajectories, trajectories_base_dir,
                          agent_dir, env.action_space.n)
        trajectories = []
        wandb.finish()

    env.close()


if __name__ == '__main__':
    record()
    print('Hello world')

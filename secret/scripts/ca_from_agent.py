import numpy as np
import gym
import torch
import click
import os
import subprocess
import random
from xvfbwrapper import Xvfb

import wandb
from itertools import count
from secret.src.timing import Timing

from gym_minigrid.wrappers import RGBImgObsWrapper
from secret.src.agents.dqn_pfrl import DQN_PFRL_ACTOR
from tqdm import tqdm
from secret.envs.triggers.utils import save_trajectories

trajectories = []
test_trajectories = []

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
@click.option('--group', type=str, required=True)
@click.option('--env-name', type=str, required=True)  # e.g. MiniGrid-Triggers-3x3-T1P1-v0
@click.option('--epochs', type=int, help='for reward prediction model training', required=True)  # e.g. MiniGrid-Triggers-3x3-T1P1-v0
@click.option('--seed', type=int, default=42, help='random seed used')
@click.option('--seeds', type=int, default=5, help='number of reward prediction models')
@click.option('--log-frequency', type=int, default=5e1, help='logging frequency, episodes')
@click.option('--episodes', type=int, help='number of episodes to record for', required=True)
@click.option('--test-episodes', type=int, help='number of test episodes to record for', required=True)
@click.option('--agents-dir', type=click.Path(exists=True), help='path to agent models', required=True)
@click.option('--use-wandb/--no-wandb', default=True)
def record(note, group, env_name, epochs, seed, seeds, log_frequency, episodes, test_episodes, agents_dir, use_wandb):
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

    observation_shape = env.reset()['image'].shape

    wandb.login()
    agents_base_dir = agents_dir

    agent_dirs = []
    for root, dirs, files in os.walk(agents_base_dir):
        agent_dirs = dirs
        break

    for agent_dir in agent_dirs:

        agent_dir_base = os.path.basename(agent_dir)
        agent_group = f'{group}_{agent_dir_base}'

        wandb.init(entity='ut-rl-credit',
                   project='attentional_fw_baselines',
                   tags=['SECRET', 'verification'],
                   group=agent_group,
                   notes='Collect trajectories, Train & Eval Reward Predictors',
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
        actor.load(os.path.join(agents_base_dir, agent_dir))

        print(f'\nLoaded agent {agent_dir}...\n')

        with Xvfb():
            with actor.eval_mode():

                full_agent_dir = os.path.join(agents_base_dir, agent_dir)
                trajectories_path = os.path.join(full_agent_dir, 'trajectories', 'train.pt')
                test_trajectories_path = os.path.join(full_agent_dir, 'trajectories', 'test.pt')

                timing = dict()
                if not os.path.exists(trajectories_path):

                    print('\nRecording training trajectories...\n')
                    # collect train set
                    for i_episode in tqdm(range(1, episodes+1)):
                        trajectories.append([])

                        observation = env.reset()
                        frames = [env.render(mode='rgb_array')]
                        rewards = []

                        for i_step in count():
                            last_observation = observation['image']

                            with Timing(timing, 'time_choose_act'):
                                action = actor.act(last_observation)

                            with Timing(timing, 'time_perform_act'):
                                observation, reward, done, _ = env.step(action)

                            with Timing(timing, 'time_observe'):
                                actor.observe(
                                    observation['image'], reward, done, reset=False)

                            frames.append(env.render(mode='rgb_array'))
                            rewards.append(reward)

                            # record observation for reward-prediction model
                            step = (last_observation, action, reward, done)
                            trajectories[-1].append(step)

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

                    save_trajectories(trajectories_path, trajectories, env.action_space.n)
                else:
                    print('\nUsing existing training trajectories...\n')

                if not os.path.exists(test_trajectories_path):
                    print('\nRecording test trajectories...\n')
                    # collect test set
                    for i_episode in tqdm(range(1, test_episodes+1)):
                        test_trajectories.append([])

                        observation = env.reset()
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
                            test_trajectories[-1].append(step)

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

                    save_trajectories(test_trajectories_path, test_trajectories, env.action_space.n)
                else:
                    print('\nUsing existing test trajectories...\n')

                print('\nStarting training of reward predictors...\n')
                subprocess.run(['python', 'train_reward_predictors.py', '--group', agent_group,
                                                                        '--agent', agent_dir,
                                                                        '--epochs', str(epochs),
                                                                        '--data-path', trajectories_path,
                                                                        '--test-path', test_trajectories_path,
                                                                        '--seeds', str(seeds)])

        trajectories = []
        test_trajectories = []
        wandb.finish()

    env.close()


if __name__ == '__main__':
    record()

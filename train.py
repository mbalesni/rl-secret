import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import gym
import torch
import pickle
import os
import time
import random

import gym_minigrid.envs
from gym_minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
import wandb
import click

from xvfbwrapper import Xvfb

recording = {
    'episodes': []
}
logs = {
    'episode_durations': [],
    'episode_returns': []
}

'''According to: https://docs.wandb.ai/library/log#images-and-overlays'''
def frames_to_video(frames, fps=24):
    stacked_frames = np.array(frames).transpose((0, 3, 1, 2)) # (h,w,c) -> (t,c,h,w)
    return wandb.Video(stacked_frames, fps=fps, format="gif")

def save_experience():
    print('Saving experience...')
    os.makedirs('recordings', exist_ok=True)
    recording_name = f'recordings/agent_experience_{round(time.time())}.pt'
    with open(recording_name, 'wb') as f:
        pickle.dump(recording, f)

@click.command()
@click.option('--env_name', type=str, default='MiniGrid-Triggers-3x3-v0')
@click.option('--seed', type=int, default=42, help='random seed used')
@click.option('--log_frequency', type=int, default=5e1, help='logging frequency in episodes')
@click.option('--episodes', type=int, default=1e3, help='number of episodes to train for', required=True)
@click.option('--learning_rate', type=float, default=1e-4, help='goal learning rate')
@click.option('--buffer_size', default=1e5, type=int, help='size of the replay buffer')
@click.option('--batch_size', default=32, help='size of the replay buffer before training')
@click.option('--record', is_flag=True, required=True)
def train(env_name, seed, log_frequency, episodes, 
          learning_rate, buffer_size, batch_size, record):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    experiment = f'{env_name}_s{seed}_ep{episodes}_lr{learning_rate}_buff{buffer_size}_rec{record}'

    wandb.login()
    wandb.init(project='attentional_fw_baselines',
               entity='ut-rl-credit',
               name=experiment,
               notes="mock initial run without DQN to test w&b",
            #    tags=["baseline", "paper1"],
               config=dict(
                   env_name=env_name,
                   seed=seed,
                   log_frequency=log_frequency,
                   episodes=episodes,
                   learning_rate=learning_rate,
                   buffer_size=buffer_size,
                   batch_size=batch_size,
                   record=record,
               ))
    # Upload models at the end of training
    wandb.save(os.path.join(wandb.run.dir, "*.pt"))
    wandb.save(os.path.join(wandb.run.dir, "recordings", "*.pt"))

    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = gym.make(env_name)
    if 'MiniGrid' in env_name: # support non-MiniGrid environments
        env = ImgObsWrapper(RGBImgPartialObsWrapper(env))

    obs = env.reset()

    print('Mock Training...')
    with Xvfb() as xvfb:
        for i_episode in range(episodes):
            obs = env.reset()
            recording['episodes'].append([])
            frames = [env.render(mode='rgb_array')]
            rewards = []

            while True:
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                frames.append(env.render(mode='rgb_array'))

                rewards.append(reward)

                if should_record:
                    step = (obs, action, reward, done)
                    recording['episodes'][i_episode].append(step)
                
                if done:
                    logs['episode_returns'].append(sum(rewards))
                    logs['episode_durations'].append(len(rewards))
                    rewards = []
                    break

            if (i_episode % log_frequency) == 0:
                wandb.log({
                    'avg_episode_duration': np.mean(logs['episode_durations']),
                    'avg_episode_return': np.mean(logs['episode_returns']),
                    'video': frames_to_video(frames),
                    'episode': i_episode,
                }, )
                logs['episode_durations'] = []
                logs['episode_returns'] = []
                
        env.close()

    if should_record:
        save_experience()


if __name__ == '__main__':
    train()
    
    

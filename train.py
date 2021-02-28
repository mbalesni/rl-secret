import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import gym
import torch
import click
import pickle
import os
import time
import random
from xvfbwrapper import Xvfb
import sys

import wandb
from itertools import count
from timing import Timing
from tqdm import tqdm

import gym_minigrid.envs
from gym_minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from dqn import DQNActor, RandomActor


recording = {
    'episodes': []
}
logs = {
    'episode_durations': [],
    'episode_returns': []
}

'''According to: https://docs.wandb.ai/library/log#images-and-overlays'''
def frames_to_video(frames, fps=15):
    stacked_frames = np.array(frames).transpose((0, 3, 1, 2)) # (h,w,c) -> (t,c,w,h)
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
@click.option('--episodes', type=int, help='number of episodes to train for', required=True)
@click.option('--learning-rate', type=float, default=1e-1, help='goal learning rate')
@click.option('--buffer_size', default=5e4, type=int, help='size of the replay buffer')
@click.option('--buffer-prefill-steps', default=5e3, type=int, help='size of the prefilled replay buffer')
@click.option('--batch-size', default=32, help='training batch size')
@click.option('--record/--no-record', default=True)
@click.option('--record-random', is_flag=True)
@click.option('--use_wandb/--no-wandb', default=True)
def train(env_name, seed, log_frequency, episodes, 
          learning_rate, buffer_size, buffer_prefill_steps, 
          batch_size, record, record_random, use_wandb):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    experiment = f'{env_name}_s{seed}_ep{episodes}_lr{learning_rate}_buff{buffer_size}_rec{record}_recrand{record_random}'

    if use_wandb:
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
                    buffer_prefill_steps=buffer_prefill_steps,
                    batch_size=batch_size,
                    record=record,
                    record_random=record_random,
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

    observation_shape = env.reset().shape
    random_actor = RandomActor(*observation_shape, env.action_space.n, 
                               buffer_size=buffer_size)
    actor = DQNActor(*observation_shape, env.action_space.n, 
                     learning_rate=learning_rate, buffer_size=buffer_size, 
                     batch_size=batch_size)
    if use_wandb:
        wandb.watch(actor.policy_net)


    with Xvfb() as xvfb:

        timing = dict()               
        print('Filling buffer...')
        i_episode = 0
        for i_step in tqdm(range(buffer_prefill_steps)):
            observation, reward, done, info, action = actor.step(env, i_episode, timing)

            if record_random:
                step = (observation, action, reward, done)
                recording['episodes'][i_episode].append(step)
            
            if done:
                i_episode += 1
                observation = env.reset()
                if record_random:
                    recording['episodes'].append([])
                actor.new_episode(observation)

                if i_episode > 0 and (i_episode % log_frequency) == 0 and use_wandb:
                    wandb.log({
                        'buff_episode': i_episode,
                        **{k: v['time'] / v['count'] for k, v in timing.items()}
                    }, )
                    timing = dict()

        actor.memory = random_actor.memory
        timing = dict()               
        print('Training...')
        for i_episode in tqdm(range(episodes)):
            observation = env.reset()
            actor.new_episode(observation)
            if record:
                recording['episodes'].append([])
            frames = [env.render(mode='rgb_array')]
            rewards = []

            for i_step in count():
                observation, reward, done, info, action = actor.step(env, i_episode, timing)
                frames.append(env.render(mode='rgb_array'))
                rewards.append(reward)

                if record:
                    step = (observation, action, reward, done)
                    recording['episodes'][i_episode].append(step)
                
                if done:
                    logs['episode_returns'].append(sum(rewards))
                    logs['episode_durations'].append(len(rewards))
                    rewards = []
                    break

            if i_episode > 0 and (i_episode % log_frequency) == 0 and use_wandb:

                wandb.log({
                    'avg_episode_duration': np.mean(logs['episode_durations']),
                    'avg_episode_return': torch.mean(torch.tensor(logs['episode_returns'], device='cpu', dtype=torch.float)),
                    'video': frames_to_video(frames),
                    'episode': i_episode,
                    **{k: v['time'] / v['count'] for k, v in timing.items()}
                }, )
                logs['episode_durations'] = []
                logs['episode_returns'] = []
                timing = dict()

    if record:
        save_experience()
    if use_wandb:   
        wandb.finish()
    env.close()

if __name__ == '__main__':
    train()
    
    

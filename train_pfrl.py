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
from gym_minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper, RGBImgObsWrapper
from dqn_pfrl import DQN_PFRL_ACTOR


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

def save_experience(base_dir):
    print('Saving experience...')
    directory = os.path.join(base_dir, 'recordings')
    os.makedirs(directory, exist_ok=True)
    recording_name = f'agent_experience_{round(time.time())}.pt'
    full_path = os.path.join(directory, recording_name)
    with open(full_path, 'wb') as f:
        pickle.dump(recording, f)

@click.command()
@click.option('--note', type=str, help='message to explain how is this run different')
@click.option('--env-name', type=str, default='MiniGrid-Triggers-3x3-v0')
@click.option('--seed', type=int, default=42, help='random seed used')
@click.option('--log-frequency', type=int, default=5e1, help='logging frequency, episodes')
@click.option('--model-save-frequency', type=int, default=5e3, help='model saving frequency, episodes')
@click.option('--episodes', type=int, help='number of episodes to train for', required=True)
@click.option('--learning-rate', type=float, default=1e-1, help='goal learning rate')
@click.option('--gamma', type=float, default=0.999, help='discount factor')
@click.option('--eps-start', type=float, default=1, help='starting exploration epsilon')
@click.option('--eps-end', type=float, default=0.1, help='final exploration epsilon')
@click.option('--eps-decay', type=int, default=250_000, help='final exploration epsilon')
@click.option('--buffer-size', default=5e4, type=int, help='size of the replay buffer')
@click.option('--buffer-prefill-steps', default=5e3, type=int, help='size of the prefilled replay buffer')
@click.option('--batch-size', default=32, help='training batch size')
@click.option('--part-observ/--full-observ', default=True, help='view a part of the environment or see the full picture')
@click.option('--record/--no-record', default=True)
@click.option('--record-random', is_flag=True)
@click.option('--target-freq', type=int, default=2000, help='how frequently to update target network')
@click.option('--update-freq', type=int, default=1, help='how frequently to run optimization')
@click.option('--use-wandb/--no-wandb', default=True)
def train(note, env_name, seed, log_frequency, model_save_frequency, episodes, 
          learning_rate, gamma, eps_start, eps_end, eps_decay, 
          buffer_size, buffer_prefill_steps, batch_size, 
          part_observ, record, record_random, target_freq, update_freq, use_wandb):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    wandb.login()
    wandb.init(project='attentional_fw_baselines',
            entity='ut-rl-credit',
            notes=note,
            tags=["baseline", "SECRET", "DQN", "PFRL"],
            mode='online' if use_wandb else 'disabled',
            config=dict(
                env_name=env_name,
                seed=seed,
                log_frequency=log_frequency,
                model_save_frequency=model_save_frequency,
                episodes=episodes,
                learning_rate=learning_rate,
                gamma=gamma,
                eps_start=eps_start, 
                eps_end=eps_end, 
                eps_decay=eps_decay,
                buffer_size=buffer_size,
                buffer_prefill_steps=buffer_prefill_steps,
                batch_size=batch_size,
                part_observ=part_observ,
                record=record,
                record_random=record_random,
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = gym.make(env_name)
    if 'MiniGrid' in env_name: # support non-MiniGrid environments
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

    with Xvfb() as xvfb:

        timing = dict()               
        print('Training...')
        for i_episode in range(1, episodes+1):
            observation = env.reset()
            if record:
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

                if record:
                    step = (observation, action, reward, done)
                    recording['episodes'][i_episode].append(step)
                
                if done:
                    logs['episode_returns'].append(sum(rewards))
                    logs['episode_durations'].append(len(rewards))
                    break

            if i_episode > 1 and (i_episode % log_frequency) == 0:

                avg_duration = np.mean(logs['episode_durations'])
                avg_return = torch.mean(torch.tensor(logs['episode_returns'], device='cpu', dtype=torch.float))

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

                if i_episode % model_save_frequency == 0:
                    agent_version_folder = f'ep{i_episode}_dur{avg_duration:.2f}_ret{avg_return:.2f}'
                    path_to_model = os.path.join(save_dir, 'agent', agent_version_folder)
                    os.makedirs(path_to_model, exist_ok=True)
                    actor.save(path_to_model)

                if record:
                    if use_wandb:
                        save_experience(save_dir)
                    else:
                        save_experience('recordings')
            
    env.close()
    wandb.finish()            

if __name__ == '__main__':
    train()
    
    

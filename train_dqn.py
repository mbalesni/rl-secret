import matplotlib.pyplot as plt
import numpy as np
import gym
import torch
import click
import pickle
import os
import time
import random
from xvfbwrapper import Xvfb

import wandb
from itertools import count
from tqdm import tqdm

from gym_minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper, RGBImgObsWrapper
from dqn import DQNActor, RandomActor


recording = {
    'episodes': []
}
logs = {
    'episode_durations': [],
    'episode_returns': []
}


def frames_to_video(frames, fps=15):
    '''According to: https://docs.wandb.ai/library/log#images-and-overlays'''

    stacked_frames = np.array(frames).transpose(
        (0, 3, 1, 2))  # (h,w,c) -> (t,c,w,h)
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
@click.option('--log_frequency', type=int, default=5e1, help='logging frequency in episodes')
@click.option('--episodes', type=int, help='number of episodes to train for', required=True)
@click.option('--learning-rate', type=float, default=1e-1, help='goal learning rate')
@click.option('--gamma', type=float, default=0.999, help='discount factor')
@click.option('--eps-start', type=float, default=1, help='starting exploration epsilon')
@click.option('--eps-end', type=float, default=0.1, help='final exploration epsilon')
@click.option('--eps-decay', type=int, default=250_000, help='final exploration epsilon')
@click.option('--buffer_size', default=5e4, type=int, help='size of the replay buffer')
@click.option('--buffer-prefill-steps', default=5e3, type=int, help='size of the prefilled replay buffer')
@click.option('--batch-size', default=32, help='training batch size')
@click.option('--part-observ/--full-observ', default=True, help='view a part of the environment or see the full picture')
@click.option('--record/--no-record', default=True)
@click.option('--record-random', is_flag=True)
@click.option('--use-wandb/--no-wandb', default=True)
def train(note, env_name, seed, log_frequency, episodes,
          learning_rate, gamma, eps_start, eps_end, eps_decay,
          buffer_size, buffer_prefill_steps, batch_size,
          part_observ, record, record_random, use_wandb):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    wandb.login()
    wandb.init(project='attentional_fw_baselines',
               entity='ut-rl-credit',
               notes=note,
               tags=["baseline", "SECRET", "DQN"],
               mode='online' if use_wandb else 'disabled',
               config=dict(
                   env_name=env_name,
                   seed=seed,
                   log_frequency=log_frequency,
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
               ))
    # Upload models at the end of training
    wandb.save(os.path.join(wandb.run.dir, "*.pt"))
    wandb.save(os.path.join(wandb.run.dir, "recordings", "*.pt"))

    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = gym.make(env_name)
    if 'MiniGrid' in env_name:  # support non-MiniGrid environments
        if part_observ:
            env = ImgObsWrapper(RGBImgPartialObsWrapper(env))
        else:
            env = ImgObsWrapper(RGBImgObsWrapper(env))

    observation = env.reset()
    plt.figure()
    plt.imshow(observation)
    plt.savefig('obs.png')
    observation_shape = env.reset().shape
    random_actor = RandomActor(*observation_shape, env.action_space.n,
                               buffer_size=buffer_size)
    actor = DQNActor(*observation_shape, env.action_space.n,
                     learning_rate=learning_rate, buffer_size=buffer_size,
                     gamma=gamma, batch_size=batch_size, eps_start=eps_start,
                     eps_end=eps_end, eps_decay=eps_decay)
    wandb.watch(actor.policy_net)

    with Xvfb():

        timing = dict()
        print('Filling buffer...')
        i_episode = 0
        for i_step in tqdm(range(1, buffer_prefill_steps+1)):
            observation, reward, done, info, action = random_actor.step(
                env, timing)

            if record_random:
                step = (observation, action, reward, done)
                recording['episodes'][i_episode].append(step)

            if done:
                i_episode += 1
                observation = env.reset()
                if record_random:
                    recording['episodes'].append([])
                random_actor.new_episode(observation)

                if i_episode > 1 and (i_episode % log_frequency) == 0:
                    wandb.log({
                        'buff_episode': i_episode,
                        **{k: v['time'] / v['count'] for k, v in timing.items()}
                    }, )
                    timing = dict()

        actor.memory = random_actor.memory
        timing = dict()
        print('Training...')
        for i_episode in tqdm(range(1, episodes+1)):
            observation = env.reset()
            actor.new_episode(observation)
            if record:
                recording['episodes'].append([])
            frames = [env.render(mode='rgb_array')]
            rewards = []

            for i_step in count():
                observation, reward, done, info, action = actor.step(
                    env, i_episode, timing)
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

            if i_episode > 1 and (i_episode % log_frequency) == 0:

                wandb.log({
                    'avg_episode_duration': np.mean(logs['episode_durations']),
                    'avg_episode_return': torch.mean(torch.tensor(logs['episode_returns'], device='cpu', dtype=torch.float)),
                    'video': frames_to_video(frames),
                    'episode': i_episode,
                    'agent_step': actor.steps_done,
                    'epsilon': actor.eps_threshold,
                    **{k: v['time'] / v['count'] for k, v in timing.items()}
                }, )
                logs['episode_durations'] = []
                logs['episode_returns'] = []
                timing = dict()

                torch.save(actor.policy_net.state_dict(),
                           os.path.join(wandb.run.dir, 'dqn_policy.pt'))

                if record:
                    if use_wandb:
                        save_experience(wandb.run.dir)
                    else:
                        save_experience('recordings')

    env.close()
    wandb.finish()


if __name__ == '__main__':
    train()

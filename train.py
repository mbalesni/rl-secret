import torch
import gym
import matplotlib
import cv2
import numpy as np
import argparse
import pickle
import os
import time
from gym_minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

should_record = False
recording = {
    'episodes': []
}

def step(env, action):
    obs, reward, done, info = env.step(action)

    if reward != 0:
        print(f'step: {env.step}, reward: {reward}')

    if should_record:
        step = (obs, action, reward, done)
        recording['episodes'][i_episode].append(step)

    return done

def save_experience():
    print('Saving experience...')
    os.makedirs('recordings', exist_ok=True)
    recording_name = f'recordings/agent_experience_{round(time.time())}.pt'
    with open(recording_name, 'wb') as f:
        pickle.dump(recording, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        help="gym environment to load",
        default='MiniGrid-Triggers-3x3-v0'
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=-1
    )
    parser.add_argument(
        "--record",
        help="record agent experience",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "--episodes",
        help="how many episodes to play",
        default=100,
    )
    args = parser.parse_args()

    should_record = args.record
    env_name = args.env
    n_episodes = int(args.episodes)

    env = gym.make(env_name)
    env = ImgObsWrapper(RGBImgPartialObsWrapper(env))

    obs = env.reset()
    print(obs.shape)
    try:
        for i_episode in range(n_episodes):
            obs = env.reset()
            recording['episodes'].append([])

            while True:
                env.render()
                action = env.action_space.sample()
                done = step(env, action)
                
                if done:
                    break
                
    except KeyboardInterrupt:
        env.close()

    if should_record:
        save_experience()

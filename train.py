import torch
import gym
import matplotlib.pyplot as plt
from matplotlib import animation
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

"""
Ensure you have imagemagick installed with 
sudo apt-get install imagemagick

Taken from https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553
"""
def save_frames_as_gif(frames, path='./gifs', filename='gym_animation.gif'):
    #Mess with this to change frame size
    dpi = 72.
    plt.figure(figsize=(frames[0].shape[1] / dpi, frames[0].shape[0] / dpi), dpi=dpi)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    os.makedirs(path, exist_ok=True)
    anim.save(os.path.join(path, filename), writer='imagemagick', fps=30)

def step(env, action):
    obs, reward, done, info = env.step(action)

    if reward != 0:
        print(f'step: {info["step"]}, reward: {reward}')

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
        "--gifs",
        help="how often to save a gif of agent completing an episode",
        default=-1, # -1 is never
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
    gifs_frequency = int(args.gifs)

    env = gym.make(env_name)
    env = ImgObsWrapper(RGBImgPartialObsWrapper(env))

    obs = env.reset()
    print(obs.shape)
    try:
        for i_episode in range(n_episodes):
            obs = env.reset()
            recording['episodes'].append([])
            frames = []

            while True:
                frames.append(env.render(mode='rgb_array'))
                action = env.action_space.sample()
                done = step(env, action)
                
                if done:
                    if gifs_frequency > 0 and i_episode % gifs_frequency == 0
                        save_frames_as_gif(frames, filename=f'episode_{i_episode}.gif')
                    break
                
    except KeyboardInterrupt:
        env.close()

    if should_record:
        save_experience()

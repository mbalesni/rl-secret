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
# import matplotlib.pyplot as plt
# import time

from gym_minigrid.wrappers import RGBImgBothObsWrapper
from config import PAD_VAL, SEQ_LEN, ACTION_SIZE, ATTN_THRESHOLD
from trajectories import one_hot_encode_action
from ppo_pfrl import PPO_PFRL_ACTOR

# from pfrl.experiments.hooks import StepHook
from pfrl import experiments

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def frames_to_video(frames, fps=15):
    stacked_frames = np.array(frames).transpose((0, 3, 1, 2))  # (h,w,c) -> (t,c,w,h)
    return wandb.Video(stacked_frames, fps=fps, format="gif")


def is_trigger_step(partial_observation, action, forward_action=2, trigger_color=[255, 76, 76]):
    '''
    Check if at this step the agent is activating the trigger by stepping onto it.

    partial_observation - numpy array of shape (H, W, C)
    action - tensor of shape (A)
    '''

    H, W, _ = partial_observation.shape
    center_h = H // 2
    center_w = W // 2

    center = partial_observation[center_h, center_w, :]
    trigger_ahead = (center == np.array(trigger_color)).all()
    going_forward = action == forward_action

    return trigger_ahead and going_forward


def act_id_to_str(act_id):
    return {0: 'left', 1: 'right', 2: 'forward'}[act_id]


def redistribute_reward(agent, final_step, reward_steps, trigger_steps_gt,
                        ca_model=None, observations=None, actions=None, rewards=None):
    '''Redistribute reward using ground truth.'''

    memories = agent.memory.memory.data

    # if not (len(trigger_steps_gt) > 0 and len(reward_steps) > 0):
    #     return

    # print()
    # print('Final step:', final_step)

    if ca_model:

        # calculate attention for the last episode

        observations_cuda = observations.transpose(2, 4).to(device)
        actions_cuda = actions.to(device)

        output, attention_output = ca_model(observations_cuda, actions_cuda, output_attention=True)
        output = output.permute(0, 2, 1)
        pred_rewards = output.argmax(dim=1)
        pred_rewards = pred_rewards.transpose(0, 1).cpu()

        # for each non-zero reward in episode,
        # identify causal steps
        # and to each causal step
        # add the reward
        for reward_step in reward_steps:
            reward_mem_idx = reward_step - final_step - 1
            memory = memories[reward_mem_idx][0]
            reward = memory['reward']
            pred_reward = pred_rewards[reward_step].item()-1

            # redistribute only if prediction is correct
            if reward != pred_reward:
                continue

            attention_vals = attention_output[0, reward_step]
            attention_discrete = torch.gt(attention_vals, ATTN_THRESHOLD, out=torch.empty(attention_vals.shape, dtype=torch.uint8, device=device))
            causal_steps = torch.where(attention_discrete == 1)[0]

            for causal_step in causal_steps:
                causal_mem_idx = causal_step - final_step - 1

                # plt.imshow(observations[causal_step].type(torch.int32).squeeze())
                # plt.savefig(f'{time.time()}_gt_act{torch.argmax(actions[causal_step]).item()}.png')
                # plt.close()

                memories[causal_mem_idx][0]['reward'] += reward

    else:
        for reward_step in reward_steps:
            reward_mem_idx = -(final_step - reward_step + 1)
            memory = memories[reward_mem_idx][0]
            reward = memory['reward']

            for trigger_step in trigger_steps_gt:
                # print('trigger step:', trigger_step)
                trigger_mem_idx_gt = -(final_step - trigger_step + 1)
                memories[trigger_mem_idx_gt][0]['reward'] += reward

    # for i in reversed(range(1, final_step+2)):
    #     index = -i
    #     obs = memories[index][0]['state']
    #     reward = memories[index][0]['reward']
    #     action = memories[index][0]['action']
    #     plt.imshow(obs)
    #     step = final_step+1 - i
    #     plt.savefig(f'debug/step{step}_{act_id_to_str(action)}_rew{reward}')

    # print('done saving')


def run_training(episodes, agent, env, obs_shape, save_path, log_frequency, learning_rate=None,
                 learning_rate_decay_steps=None, use_redistributed_reward=False, ca_model=None, log_prefix='', gifs=False):
    '''Run training of a PPO PFRL agent with SECRET attention model and custom reward redistribution.'''

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

    lr_scheduler = experiments.LinearInterpolationHook(learning_rate_decay_steps, learning_rate, 0, lr_setter)
    global_timestep = 0

    with Xvfb():
        for i_episode in tqdm(range(1, episodes+1), mininterval=0.5):

            observation = env.reset()
            rewards = []

            if gifs:
                frames = [env.render(mode='rgb_array')]

            if use_redistributed_reward:
                reward_steps = []
                trigger_steps_gt = []

                part_obs_shape = observation['image_partial'].shape
                ca_partial_observations = torch.zeros((SEQ_LEN, 1, *part_obs_shape), dtype=torch.float32)
                ca_actions = torch.ones((SEQ_LEN, 1, ACTION_SIZE), dtype=torch.float32) * int(PAD_VAL)
                ca_rewards = torch.ones((SEQ_LEN, 1), dtype=torch.long) * int(PAD_VAL)

            for i_step in count():
                global_timestep += 1
                last_obs = observation['image_partial']

                with Timing(timing, 'time_choose_act'):
                    action = agent.act(observation['image'])

                with Timing(timing, 'time_perform_act'):
                    observation, reward, done, _ = env.step(action)

                    if use_redistributed_reward:
                        # for ground-truth based causality, redistribute only positive reward
                        if (ca_model is None and reward > 0) or (ca_model is not None and abs(reward) > 0):
                            reward_steps.append(i_step)

                        ca_partial_observations[i_step] = torch.tensor(last_obs, dtype=torch.float32)
                        ca_actions[i_step] = one_hot_encode_action(action, ACTION_SIZE).type(torch.float32)
                        ca_rewards[i_step] = reward + 1  # reward class (0,1,2)

                        if is_trigger_step(last_obs, action):
                            trigger_steps_gt.append(i_step)

                with Timing(timing, 'time_observe'):
                    agent.observe(observation['image'], reward, done, reset=False)
                    lr_scheduler(env, agent, global_timestep)

                if gifs:
                    frames.append(env.render(mode='rgb_array'))
                rewards.append(reward)

                if done:
                    logs['episode_returns'].append(sum(rewards))
                    logs['episode_durations'].append(len(rewards))
                    break

            if use_redistributed_reward:
                with Timing(timing, 'time_redistribute_reward'):
                    # TODO: make work for PPO
                    redistribute_reward(agent, i_step, reward_steps, trigger_steps_gt, ca_model, ca_partial_observations, ca_actions, ca_rewards)

            if i_episode > 1 and (i_episode % log_frequency) == 0:

                avg_duration = np.mean(logs['episode_durations'])
                avg_return = torch.mean(torch.tensor(logs['episode_returns'], device='cpu', dtype=torch.float))
                agent_stats = agent.get_statistics()

                wandb_payload = ({
                    f'{log_prefix}avg_episode_duration': avg_duration,
                    f'{log_prefix}avg_episode_return': avg_return,
                    f'{log_prefix}episode': i_episode,
                    **{f'{log_prefix}{k}': v['time'] / v['count'] for k, v in timing.items()},
                    **{f'{log_prefix}ppo_{k}': v for k, v in agent_stats},
                })
                if gifs:
                    wandb_payload[f'{log_prefix}video'] = frames_to_video(frames)
                wandb.log(wandb_payload)
                logs['episode_durations'] = []
                logs['episode_returns'] = []
                timing = dict()

                if avg_return > best_return:
                    best_return = avg_return
                    agent.save(best_agent_dir)


# Linearly decay the learning rate to zero
def lr_setter(env, agent, value):
    for param_group in agent.optimizer.param_groups:
        param_group["lr"] = value


@click.command()
@click.option('--batch-size', default=32, help='training batch size')
@click.option('--ca-model-path', type=click.Path(exists=True), help='path to a creadit assignment model for reward redistribution')
@click.option('--clip-eps', type=float, default=0.2, help='epsilon clipping')
@click.option('--entropy-coef', type=float, default=0.01, help='weight for ppo entropy loss term')
@click.option('--env-name', type=str, default='MiniGrid-Triggers-3x3-T1P1-v0')
@click.option('--episodes', type=int, help='number of episodes to train for', required=True)
@click.option('--epochs', type=int, default=10, help='epochs to run a PPO training iteration for')
@click.option('--gamma', type=float, default=0.98, help='discount factor')
@click.option('--gifs/--no-gifs', default=False, help='whether to save some episodes as gifs to W&B')
@click.option('--group', type=str)
@click.option('--log-frequency', type=int, default=5e1, help='logging frequency, episodes')
@click.option('--learning-rate', type=float, default=0.00025, help='goal learning rate')
@click.option('--learning-rate-decay', type=int, default=1e7, help='steps to decay lr to 0')
@click.option('--note', type=str, help='message to explain how is this run different')
@click.option('--seed', type=int, default=42, help='random seed used')
@click.option('--update-freq', type=int, default=1024, help='how frequently to run optimization')
@click.option('--use-wandb/--no-wandb', default=True)
@click.option('--use-redistributed-reward/--vanilla', default=False, help='whether to use causal reward redistribution')
def main(batch_size,
         ca_model_path, clip_eps,
         entropy_coef, env_name, episodes, epochs,
         gamma, gifs, group,
         log_frequency, learning_rate, learning_rate_decay,
         note,
         seed,
         update_freq, use_wandb, use_redistributed_reward):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tags = ['ppo']
    if use_redistributed_reward:
        tags.append('SECRET')
    else:
        tags.append('vanilla-rl')

    wandb.login()
    wandb.init(entity='ut-rl-credit',
               project='attentional_fw_baselines',
               group=group,
               notes=note,
               tags=tags,
               mode='online' if use_wandb else 'disabled',
               config=dict(
                   batch_size=batch_size,
                   ca_model_path=ca_model_path,
                   clip_eps=clip_eps,
                   entropy_coef=entropy_coef,
                   env_name=env_name,
                   episodes=episodes,
                   epochs=epochs,
                   gamma=gamma,
                   learning_rate=learning_rate,
                   learning_rate_decay=learning_rate_decay,
                   log_frequency=log_frequency,
                   seed=seed,
                   update_freq=update_freq,
                   use_redistributed_reward=use_redistributed_reward,
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
        env = RGBImgBothObsWrapper(env)

    observation_obj = env.reset()
    observation_shape = observation_obj['image'].shape
    agent = PPO_PFRL_ACTOR(*observation_shape, env.action_space.n,
                           learning_rate=learning_rate, learning_rate_decay_steps=learning_rate_decay,
                           gamma=gamma, batch_size=batch_size,
                           update_interval=update_freq, entropy_coef=entropy_coef, env=env, epochs=epochs)

    ca_model = None

    if ca_model_path:
        ca_model = torch.load(ca_model_path).to(device)

    run_training(episodes, agent, env, observation_shape, agent_save_dir, log_frequency, use_redistributed_reward=use_redistributed_reward,
                 learning_rate=learning_rate, learning_rate_decay_steps=learning_rate_decay, ca_model=ca_model, gifs=gifs)

    env.close()
    wandb.finish()


if __name__ == '__main__':
    main()

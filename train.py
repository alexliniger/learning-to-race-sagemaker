import argparse
import json
import os
import sagemaker_containers
import sys
from gym_racecar.envs import race_car_center_line_env
from gym_racecar.envs import race_car_center_line_realistic_env

import gym
import numpy as np
import torch
import wandb_callback

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


import subprocess
import sys
from pathlib import Path


def train(args,config):
    log_dir = 'logs'
    training_steps = config["steps"]
    action_obs = config["action_obs"]

    logs_dir = Path('logs')
    logs_dir.mkdir(parents=True, exist_ok=True)

    # env = gym.make('Pendulum-v0')
    # eval_env = gym.make('Pendulum-v0')
    # hard_eval_env = gym.make('Pendulum-v0')

    env = gym.make('gym_racecar:RaceCar-v0', action_obs=action_obs)
    eval_env = gym.make('gym_racecar:RaceCar-v0', action_obs=action_obs)
    hard_eval_env = gym.make('gym_racecar:RaceCarRealistic-v0', action_obs=action_obs)

    env = Monitor(env, log_dir)

    _wandb_callback = wandb_callback.WandbCallback(eval_env=eval_env,hard_eval_env=hard_eval_env,
                                                   best_model_save_path='./logs/',
                                                   log_path='./logs/',
                                                   eval_freq=config["eval_freq"], train_freq=config["rollout_steps"],
                                                   n_eval_episodes=10, n_final_eval_episodes=50,
                                                   deterministic=True, render=False, video=True,
                                                   wandb_name=config["wandb_name"], config=config)

    if config["activation_fn"] == "relu":
        policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=config["net_arch"])
    elif config["activation_fn"] == "tanh":
        policy_kwargs = dict(activation_fn=torch.nn.Tanh, net_arch=config["net_arch"])
    else:
        policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=config["net_arch"])
        print("activation does not match relu or tanh, used relu by default")

    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=0, learning_rate=config["lr"],
                tensorboard_log="./tensorboard/", n_steps=config["rollout_steps"], n_epochs=config["n_epochs"],
                batch_size=config["batch_size"],gamma=config["gamma"])

    model.learn(total_timesteps=training_steps, callback=[_wandb_callback])
    model.save('./logs/final_model')

    del model
    del env
    del eval_env
    del hard_eval_env



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')

    # Container environment
    parser.add_argument('--hosts', type=list, default=json.jsonloads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    #     parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    #     parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    config = json.load(open('config.json'))

    train(parser.parse_args(), config)
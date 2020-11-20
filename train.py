#!/usr/bin/env python

import argparse
import json
import os
import sagemaker_containers
import sys

import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb_callback

from stable_baselines3 import PPO

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter


def train(args):

    log_dir = './logs/'
    training_steps = 500000
    n_state = 6
    n_action = 2
    action_obs = True
    if action_obs:
        n_obs = 9
    else:
        n_obs = 7

    env = gym.make('gym_racecar:RaceCar-v0', action_obs=action_obs)
    eval_env = gym.make('gym_racecar:RaceCar-v0', action_obs=action_obs)
    env = Monitor(env, log_dir)

    wandb_callback = wandb_callback.WandbCallback(eval_env, best_model_save_path='./logs/',
                                                  log_path='./logs/', eval_freq=10000,
                                                  deterministic=True, render=False)
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[256, 256, 256])
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4, tensorboard_log="./race_car_tensorboard/")
    model.learn(total_timesteps=training_steps, callback=[wandb_callback])
    model.save('./logs/final_model')

    del model
    del env
    del eval_env



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')

    # Container environment
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    train(parser.parse_args())
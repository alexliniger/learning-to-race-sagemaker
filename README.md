# DLAD Exercise learning-to-race
This is the base script for the RL exercise of the Deep Learning for Autonomous Driving (DLAD) lecture at CVL - ETH Zurich

## Goal
The main goal of the exercise is to investigate different RL algorithms and to understand their advantages and drawbacks for an autonomous driving application. The task is to learn an autonomous racing policy for a 1:43 scale RC car.

## What is included in the REPO

The repo includes all the base functionalities to train and log the training of an RL agent with Stable Baselines 3.
The two main parts of the repo is a logging script using WandB (https://wandb.ai), and the integration with AWS Sagemaker (https://aws.amazon.com/sagemaker/) to train the RL agent in the cloud using SPOT instance pricing.

## What to expect (from base implementation)
The base script uses PPO from stable baselines with default parameters. The RL agent learns to drive but cannot achieve good performance causing a significant amount of crashes as well as not driving very fast. Classical methods using Model Predictive Control (https://github.com/alexliniger/MPCC) achieve by far superior driving performance.

### After 10k steps
<img src="https://github.com/alexliniger/learning-to-race-sagemaker/blob/main/video/output_10k.gif" width="300" />

### After 500k steps
<img src="https://github.com/alexliniger/learning-to-race-sagemaker/blob/main/video/output_500k.gif" width="300" />

### After 1M steps
<img src="https://github.com/alexliniger/learning-to-race-sagemaker/blob/main/video/output_1M.gif" width="300" />

### After 2M steps
<img src="https://github.com/alexliniger/learning-to-race-sagemaker/blob/main/video/output_2M.gif" width="300" />

### After 3M steps
<img src="https://github.com/alexliniger/learning-to-race-sagemaker/blob/main/video/output_3M.gif" width="300" />

## Tasks
- Investigate the learning behavior of PPO and investigate methods to improve the performance?
- Compare the performance with other RL algorithms such as SAC? Does the sample complexity change?
- Investigate how the performance changes when you use the realistic (with noise) gym? Which algorithm does a better transfer to this more realistic task?
- Investigate if the policy learned in the standard gym can be used to speed up the training on the more complex realistic case?
- Do the findings you learned on the simpler experiments transfer to the hard case where you do not have access to the ideal line?

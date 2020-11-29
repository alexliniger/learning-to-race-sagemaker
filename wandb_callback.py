import numpy as np
import gym
import wandb

from stable_baselines3.common.callbacks import BaseCallback
from gym.wrappers.monitoring.video_recorder import ImageEncoder
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from typing import Any, Dict, Optional, Union, Tuple
from stable_baselines3 import PPO
import os
import warnings

"""
    Callback for evaluating an agent.

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every eval_freq call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param deterministic: Whether to render or not the environment during evaluation
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

class WandbCallback(BaseCallback):
    def __init__(
            self,
            eval_env: Union[gym.Env, VecEnv],
            hard_eval_env: Union[gym.Env, VecEnv],
            n_eval_episodes: int = 5,
            n_final_eval_episodes: int = 20,
            eval_freq: int = 10000,
            train_freq: int = 2048,
            log_path: str = None,
            best_model_save_path: str = None,
            deterministic: bool = True,
            render: bool = False,
            video: bool = False,
            wandb_name: str = "test",
            config: dict = {},
            warn: bool = True,
    ):
        super(WandbCallback, self).__init__(verbose=1)
        self.n_eval_episodes = n_eval_episodes
        self.n_final_eval_episodes = n_final_eval_episodes
        self.eval_freq = eval_freq
        self.train_freq = train_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.video = video
        self.n_inputs = len(eval_env.action_space.shape)

        self.steps = 0

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])
        if not isinstance(hard_eval_env, VecEnv):
            hard_eval_env = DummyVecEnv([lambda: hard_eval_env])

        if isinstance(eval_env, VecEnv):
            assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

        self.eval_env = eval_env
        self.hard_eval_env = hard_eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []

        wandb.init(project=wandb_name,config=config)


    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        wandb.watch(self.model.policy, log="all", log_freq=100)

    def _on_step(self) -> bool:

        if self.train_freq > 0 and self.n_calls % self.train_freq == 0:
            self.write(self.logger.Logger.CURRENT.name_to_value, self.logger.Logger.CURRENT.name_to_excluded)

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            wandb.log({"eval/mean_reward": float(mean_reward)},step=self.model.num_timesteps)
            wandb.log({"eval/mean_ep_length": mean_ep_length},step=self.model.num_timesteps)
            wandb.log({"eval/reward_hist": wandb.Histogram(episode_rewards)},step=self.model.num_timesteps)
            wandb.log({"eval/ep_length_hist": wandb.Histogram(episode_lengths)},step=self.model.num_timesteps)

            if mean_reward > self.best_mean_reward:
                print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward

            eval_video_path = 'logs/videos/eval.mp4'
            video_length = 400

            input_dict = {}
            for m in range(self.n_inputs):
                input_dict[m] = []

            obs = self.eval_env.reset()
            if self.video:
                # Record the video starting at the first step
                list_render = []
                list_render.append(self.eval_env.render(mode='rgb_array'))

            for _ in range(video_length + 1):
                action, _ = self.model.predict(obs)
                for m in range(self.n_inputs):
                    input_dict[m].append(action[0][m])

                obs, _, _, _ = self.eval_env.step(action)
                if self.video:
                    list_render.append(self.eval_env.render(mode='rgb_array'))

            if self.video:
                encoder = ImageEncoder(eval_video_path, list_render[0].shape, 30, 30)
                for im in list_render:
                    encoder.capture_frame(im)
                encoder.close()

                wandb.log({f'video/{self.model.num_timesteps}': wandb.Video(eval_video_path)},
                          step=self.model.num_timesteps)

            # log histogram with inputs
            for m in range(self.n_inputs):
                wandb.log({"action_eval/input"+str(m)+"_box_hist": wandb.Histogram(input_dict[m])}, step=self.model.num_timesteps)

        return True

    def _on_training_end(self) -> None:

        self._final_eval(self.model,self.eval_env,'final')
        self._final_eval(self.model, self.hard_eval_env, 'final_hard')

        model = PPO.load(os.path.join(self.best_model_save_path, "best_model"))
        self._final_eval(model, self.eval_env, 'best')
        self._final_eval(model, self.hard_eval_env, 'best_hard')

    def _final_eval(self,model,eval_env,mode):
        # Sync training and eval env if there is VecNormalize
        sync_envs_normalization(self.training_env, eval_env)

        episode_rewards, episode_lengths = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=self.n_final_eval_episodes,
            render=self.render,
            deterministic=self.deterministic,
            return_episode_rewards=True,
        )

        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
        self.last_mean_reward = mean_reward

        print("Eval " + mode + f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
        # Add to current Logger
        wandb.log({mode+"/mean_reward": float(mean_reward)}, step=self.model.num_timesteps)
        wandb.log({mode+"/mean_ep_length": mean_ep_length}, step=self.model.num_timesteps)

        data_r = [[r[0]] for r in episode_rewards]
        data_l = [[float(l)+1e-9] for l in episode_lengths]
        table_reward = wandb.Table(data=data_r, columns=["reward"])
        wandb.log({mode+"/reward_hist": wandb.plot.histogram(table_reward, value="reward")})
        table_ep_length = wandb.Table(data=data_l, columns=["length"])
        wandb.log({mode + "/ep_length_hist": wandb.plot.histogram(table_ep_length, value="length")})

        eval_video_path = 'logs/videos/'+mode+'_eval.mp4'
        video_length = 400
        input_dict = {}
        for m in range(self.n_inputs):
            input_dict[m] = []

        obs = self.eval_env.reset()
        if self.video:
            # Record the video starting at the first step
            list_render = []
            list_render.append(self.eval_env.render(mode='rgb_array'))

        for _ in range(video_length + 1):
            action, _ = self.model.predict(obs)
            for m in range(self.n_inputs):
                input_dict[m].append(action[0][m])

            obs, _, _, _ = self.eval_env.step(action)
            if self.video:
                list_render.append(self.eval_env.render(mode='rgb_array'))

        if self.video:
            encoder = ImageEncoder(eval_video_path, list_render[0].shape, 30, 30)
            for im in list_render:
                encoder.capture_frame(im)
            encoder.close()

            wandb.log({mode+"/video": wandb.Video(eval_video_path)}, step=self.model.num_timesteps)

        # log histogram with inputs
        for m in range(self.n_inputs):
            wandb.log({mode+"/input" + str(m) + "_box_hist": wandb.Histogram(input_dict[m])},
                      step=self.model.num_timesteps)





    def write(self, key_values: Dict[str, Any], key_excluded: Dict[str, Union[str, Tuple[str, ...]]]) -> None:

        for (key, value), (_, excluded) in zip(sorted(key_values.items()), sorted(key_excluded.items())):

            if excluded is not None and "tensorboard" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                wandb.log({key:value},step=self.model.num_timesteps)



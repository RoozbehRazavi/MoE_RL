import imageio
import torch
import os
import re
import yaml
import csv
import logging
from logging import Handler
import shutil


def create_directory(dir_path):
    # Check if the directory exists
    if os.path.exists(dir_path):
        return

    # Create the directory again
    os.makedirs(dir_path)
    print(f"Created new directory: {dir_path}")


def recreate_directory(dir_path):
    # Check if the directory exists
    if os.path.exists(dir_path):
        # Remove the directory and its contents
        shutil.rmtree(dir_path)
        print(f"Removed existing directory: {dir_path}")

    # Create the directory again
    os.makedirs(dir_path)
    print(f"Created new directory: {dir_path}")


def save_frames_as_video(frames, path, name):
    writer = imageio.get_writer(f'{path}/{name}.gif', fps=20)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def pad_sequences(sequences, max_len, dim, batch_size):
    padded = torch.full((batch_size, max_len, dim), 0, dtype=torch.float32)
    for i, batch in enumerate(sequences):
        for j, transition in enumerate(batch):
            padded[i, j] = torch.tensor(transition)
    return padded


def find_latest_checkpoint(directory):
    # This pattern matches "model_checkpoint" followed by any number (the iteration), and ends with ".pth"
    pattern = re.compile(r'model_checkpoint(\d+)\.pth$')

    highest_iteration = -1
    latest_file = None

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # Check if the filename matches the specified format
        match = pattern.match(filename)
        if match:
            # Extract the iteration number and convert to an integer
            iteration = int(match.group(1))
            # If this iteration is higher than the current highest, update the highest and latest file
            if iteration > highest_iteration:
                highest_iteration = iteration
                latest_file = filename

    # Return the path to the file with the highest iteration
    if latest_file:
        return os.path.join(directory, latest_file)
    else:
        return None


def save_config_to_yaml(args, filepath):
    with open(filepath, 'w') as yaml_file:
        # Convert args namespace to a dictionary before serialization
        yaml.dump(vars(args), yaml_file, default_flow_style=False)


def load_config_from_yaml(filepath):
    with open(filepath, 'r') as yaml_file:
        config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return config_dict


class RLCSVLogHandler(Handler):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.fieldnames = ['episode', 'time_step', 'return']
        # Ensure the file is set up with headers if it's new
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            if csvfile.tell() == 0:  # Write header only if file is empty
                writer.writeheader()

    def emit(self, record):
        # Assume record.message is a list of (time_step, return) tuples
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            for time_step, ret in record.message:
                writer.writerow({'episode': record.episode, 'time_step': time_step, 'return': ret})


class ListLogFormatter(logging.Formatter):
    def format(self, record):
        # This custom format expects the 'episode' to be passed in extra
        record.episode = record.args['episode']
        record.message = record.args['returns']
        return super().format(record)


def setup_rl_logger(path):
    logger = logging.getLogger('RLLogger')
    logger.setLevel(logging.INFO)  # Typically, we log info-level in such cases
    handler = RLCSVLogHandler(f'{path}/rl_returns.csv')
    handler.setFormatter(ListLogFormatter())
    logger.addHandler(handler)
    return logger

from stable_baselines3.common.buffers import BaseBuffer
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize
from typing import NamedTuple
from stable_baselines3.common.callbacks import BaseCallback
try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None



class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    rewards: th.Tensor

class RecurrentReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        episode_max_len: int,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.episode_max_len = episode_max_len 
        self.episode_counter = np.zeros((self.n_envs))
        self.episode_lens = np.zeros((self.buffer_size))
        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.episode_max_len, *self.obs_shape), dtype=observation_space.dtype)

        self.temp_observations = np.zeros((self.n_envs, self.episode_max_len, *self.obs_shape), dtype=observation_space.dtype) 

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_observations = np.zeros((self.buffer_size, self.episode_max_len, *self.obs_shape), dtype=observation_space.dtype)

            self.temp_next_observations = np.zeros((self.n_envs, self.episode_max_len, *self.obs_shape), dtype=observation_space.dtype) 

        self.actions = np.zeros(
            (self.buffer_size, self.episode_max_len, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )

        self.temp_actions = np.zeros((self.n_envs, self.episode_max_len, self.action_dim), self._maybe_cast_dtype(action_space.dtype)) 

        self.rewards = np.zeros((self.buffer_size, self.episode_max_len), dtype=np.float32)
        
        self.temp_rewards = np.zeros((self.n_envs, self.episode_max_len), dtype=np.float32) 
        
        self.dones = np.zeros((self.buffer_size, self.episode_max_len), dtype=np.float32)
        
        self.temp_dones = np.zeros((self.n_envs, self.episode_max_len), dtype=np.float32) 
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.episode_max_len), dtype=np.float32)

        self.temp_timeouts = np.zeros((self.n_envs, self.episode_max_len), dtype=np.float32) 

        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            )

            if not optimize_memory_usage:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.temp_observations[:, self.episode_counter] = np.array(obs)

        if self.optimize_memory_usage:
            self.temp_next_observations[:, self.episode_counter + 1] = np.array(next_obs)

        else:
            self.temp_next_observations[:, self.episode_counter] = np.array(next_obs)

        self.temp_actions[:, self.episode_counter] = np.array(action)
        self.temp_rewards[:, self.episode_counter] = np.array(reward)
        self.temp_dones[:, self.episode_counter] = np.array(done)

        if self.handle_timeout_termination:
            self.temp_timeouts[:, self.episode_counter] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.episode_counter += 1
        
        if sum(done) > 0:
            self.observations[self.pos:self.pos + sum(done)] = self.temp_observations[done]
            self.next_observations[self.pos:self.pos + sum(done)] = self.temp_next_observations[done]
            self.actions[self.pos:self.pos + sum(done)] = self.temp_actions[done]
            self.rewards[self.pos:self.pos + sum(done)] = self.temp_rewards[done]
            self.dones[self.pos:self.pos + sum(done)] = self.temp_dones[done]
            self.timeouts[self.pos:self.pos + sum(done)] = self.temp_timeouts[done]

            self.pos += sum(done)
            self.episode_lens[self.pos: self.pos:sum()] = self.episode_counter[done]
            
            self.episode_counter[done] = 0
            self.temp_observations[done] = np.zeros((sum(done), self.episode_max_len, *self.obs_shape), dtype=self.observation_space.dtype) 
            self.temp_next_observations[done] = np.zeros((sum(done), self.episode_max_len, *self.obs_shape), dtype=self.observation_space.dtype) 
            self.temp_actions[done] = np.zeros((sum(done), self.episode_max_len, self.action_dim), self._maybe_cast_dtype(self.action_space.dtype)) 
            self.temp_rewards[done] = np.zeros((sum(done), self.episode_max_len), dtype=np.float32)
            self.temp_dones[done] = np.zeros((sum(done), self.episode_max_len), dtype=np.float32)
            self.temp_timeouts[done] = np.zeros((sum(done), self.episode_max_len), dtype=np.float32)
        
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, :, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, :, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, :, :], env),
            self.actions[batch_inds, :, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            self.dones[batch_inds, :],
            self._normalize_reward(self.rewards[batch_inds, :], env),
            self.episode_lens[batch_inds]
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
        """
        Cast `np.float64` action datatype to `np.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype

class CustomCallback(BaseCallback):
    def __init__(self, model, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.model = model

    def _on_rollout_start(self) -> None:
        self.model.reset_hidden_state()

    def _on_step(self) -> bool:
        return True
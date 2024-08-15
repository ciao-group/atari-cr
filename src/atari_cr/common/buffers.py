import copy
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from stable_baselines3.common.buffers import BaseBuffer, ReplayBuffer

from atari_cr.common.utils import get_action_dim, get_obs_shape, get_device

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


TensorDict = Dict[Union[str, int], th.Tensor]
class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class DictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor


class DictReplayBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    next_observations: TensorDict
    dones: th.Tensor
    rewards: th.Tensor

class DoubleActionReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    motor_actions: th.Tensor
    sensory_actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor

class DoubleActionWithFovlocReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    fov_locs: th.Tensor
    motor_actions: th.Tensor
    sensory_actions: th.Tensor
    next_observations: th.Tensor
    next_fov_locs: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor

class DoubleActionReplayBuffer(BaseBuffer):
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

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        motor_action_space: spaces.Space,
        sensory_action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.motor_action_space = motor_action_space
        self.sensory_action_space = sensory_action_space
        self.obs_shape = get_obs_shape(observation_space)

        self.motor_action_dim = get_action_dim(motor_action_space)
        self.sensory_action_dim = get_action_dim(sensory_action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

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

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)

        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)

        self.motor_actions = np.zeros((self.buffer_size, self.n_envs, self.motor_action_dim), dtype=motor_action_space.dtype)
        self.sensory_actions = np.zeros((self.buffer_size, self.n_envs, self.sensory_action_dim), dtype=sensory_action_space.dtype)

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.infos = [None] * self.buffer_size

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage = self.observations.nbytes + self.motor_actions.nbytes + self.sensory_actions.nbytes + self.rewards.nbytes + self.dones.nbytes

            if self.next_observations is not None:
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
        motor_action: np.ndarray,
        sensory_action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: Dict[str, Any],
    ) -> None:

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)
            next_obs = next_obs.reshape((self.n_envs,) + self.obs_shape)

        # Same, for actions
        motor_action = motor_action.reshape((self.n_envs, self.motor_action_dim))
        sensory_action = sensory_action.reshape((self.n_envs, self.sensory_action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.motor_actions[self.pos] = np.array(motor_action).copy()
        self.sensory_actions[self.pos] = np.array(sensory_action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.infos[self.pos] = copy.deepcopy(infos)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([False])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VectorEnv] = None) -> DoubleActionReplayBufferSamples:
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

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VectorEnv] = None) -> ReplayBufferSamples:
        """
        :param Array[batch_size] batch_inds: A batch of indices into the buffer to sample a batch of data points
        """
        # Sample randomly the env indices, indicating from which env to sample in the vec env
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.motor_actions[batch_inds, env_indices, :],
            self.sensory_actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )

        return DoubleActionReplayBufferSamples(*tuple(map(self.to_torch, data)))


class DoubleActionWithFovlocReplayBuffer(BaseBuffer):
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

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        motor_action_space: spaces.Space,
        sensory_action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        fov_loc_size: int = 2
    ):
        
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.motor_action_space = motor_action_space
        self.sensory_action_space = sensory_action_space
        self.obs_shape = get_obs_shape(observation_space)
        self.fov_loc_size = fov_loc_size

        self.motor_action_dim = get_action_dim(motor_action_space)
        self.sensory_action_dim = get_action_dim(sensory_action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

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

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)
        self.fov_locs = np.zeros((self.buffer_size, self.n_envs) + self.fov_loc_size, dtype=np.float32)
        

        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
            self.next_fov_locs = None
        else:
            self.next_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)
            self.next_fov_locs = np.zeros((self.buffer_size, self.n_envs) + self.fov_loc_size, dtype=np.float32)

        self.motor_actions = np.zeros((self.buffer_size, self.n_envs, self.motor_action_dim), dtype=motor_action_space.dtype)
        self.sensory_actions = np.zeros((self.buffer_size, self.n_envs, self.sensory_action_dim), dtype=sensory_action_space.dtype)

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.infos = [None] * self.buffer_size
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage = self.observations.nbytes + self.motor_actions.nbytes + self.sensory_actions.nbytes + self.rewards.nbytes + self.dones.nbytes

            if self.next_observations is not None:
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
        fov_locs: np.ndarray,
        next_fov_locs: np.ndarray,
        motor_action: np.ndarray,
        sensory_action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: Dict[str, Any],
    ) -> None:

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)
            next_obs = next_obs.reshape((self.n_envs,) + self.obs_shape)
            fov_locs = fov_locs.reshape((self.n_envs,) + self.fov_loc_size)
            next_fov_locs = next_fov_locs.reshape((self.n_envs,) + self.fov_loc_size)

        # Same, for actions
        motor_action = motor_action.reshape((self.n_envs, self.motor_action_dim))
        sensory_action = sensory_action.reshape((self.n_envs, self.sensory_action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()
        # print (fov_locs.shape, self.fov_locs.shape)
        self.fov_locs[self.pos] = np.array(fov_locs).copy()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
            self.fov_locs[(self.pos + 1) % self.buffer_size] = np.array(next_fov_locs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()
            self.next_fov_locs[self.pos] = np.array(next_fov_locs).copy()

        self.motor_actions[self.pos] = np.array(motor_action).copy()
        self.sensory_actions[self.pos] = np.array(sensory_action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.infos[self.pos] = copy.deepcopy(infos)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([False])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VectorEnv] = None) -> DoubleActionWithFovlocReplayBufferSamples:
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

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VectorEnv] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
            next_fov_locs = self.fov_locs[(batch_inds + 1) % self.buffer_size, env_indices, :]
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)
            next_fov_locs = self.next_fov_locs[batch_inds, env_indices, :]

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.fov_locs[batch_inds, env_indices, :],
            self.motor_actions[batch_inds, env_indices, :],
            self.sensory_actions[batch_inds, env_indices, :],
            next_obs,
            next_fov_locs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )

        return DoubleActionWithFovlocReplayBufferSamples(*tuple(map(self.to_torch, data)))
    

class NstepRewardReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    discounts: th.Tensor

class NstepRewardReplayBuffer(ReplayBuffer):
    def __init__(self, n_step_reward=3, gamma=0.99, **kwargs):
        super().__init__(**kwargs)
        self.n_step_reward = n_step_reward
        self.gamma = gamma

    def sample(self, batch_size: int, env: Optional[VectorEnv] = None) -> NstepRewardReplayBufferSamples:
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size - self.n_step_reward, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos - self.n_step_reward, size=batch_size)
        return self._get_samples(batch_inds, env=env)
    
    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VectorEnv] = None) -> NstepRewardReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )

        reward = data[4]
        discount = np.ones_like(data[4], dtype=np.float32)
        for i in range(1, self.n_step_reward):
            step_reward = self.rewards[(batch_inds+i) % self.buffer_size, env_indices].reshape(-1, 1)
            discount *= np.array([0. if x else self.gamma for x in \
                                  self.dones[(batch_inds+i) % self.buffer_size, env_indices].reshape(-1, 1)], dtype=np.float32).reshape(-1, 1)
            reward += step_reward * discount

        data = (*data[:4], reward, discount)
        
        return NstepRewardReplayBufferSamples(*tuple(map(self.to_torch, data)))
    

class NstepRewardDoubleActionReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    motor_actions: th.Tensor
    sensory_actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    discounts: th.Tensor


class NstepRewardDoubleActionReplayBuffer(DoubleActionReplayBuffer):
    def __init__(self, n_step_reward=3, gamma=0.99, **kwargs):
        super().__init__(**kwargs)
        self.n_step_reward = n_step_reward
        self.gamma = gamma

    def sample(self, batch_size: int, env: Optional[VectorEnv] = None) -> NstepRewardDoubleActionReplayBufferSamples:
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size - self.n_step_reward, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos - self.n_step_reward, size=batch_size)
        return self._get_samples(batch_inds, env=env)
    
    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VectorEnv] = None) -> NstepRewardDoubleActionReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.motor_actions[batch_inds, env_indices, :],
            self.sensory_actions[batch_inds, env_indices, :],
            next_obs,
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )

        reward = data[5]
        discount = np.ones_like(data[5], dtype=np.float32)
        for i in range(1, self.n_step_reward):
            step_reward = self.rewards[(batch_inds+i) % self.buffer_size, env_indices].reshape(-1, 1)
            discount *= np.array([0. if x else self.gamma for x in \
                                  self.dones[(batch_inds+i) % self.buffer_size, env_indices].reshape(-1, 1)], dtype=np.float32).reshape(-1, 1)
            reward += step_reward * discount

        data = (*data[:5], reward, discount)
        
        return NstepRewardDoubleActionReplayBufferSamples(*tuple(map(self.to_torch, data)))
    

class NstepRewardDoubleActionWithFovlocReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    fov_locs: th.Tensor
    motor_actions: th.Tensor
    sensory_actions: th.Tensor
    next_observations: th.Tensor
    next_fov_locs: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    discounts: th.Tensor


class NstepRewardDoubleActionWithFovlocReplayBuffer(DoubleActionWithFovlocReplayBuffer):
    def __init__(self, n_step_reward=3, gamma=0.99, **kwargs):
        super().__init__(**kwargs)
        self.n_step_reward = n_step_reward
        self.gamma = gamma

    def sample(self, batch_size: int, env: Optional[VectorEnv] = None) -> NstepRewardDoubleActionWithFovlocReplayBufferSamples:
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size - self.n_step_reward, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos - self.n_step_reward, size=batch_size)
        return self._get_samples(batch_inds, env=env)
    
    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VectorEnv] = None) -> NstepRewardDoubleActionWithFovlocReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
            next_fov_locs = self.fov_locs[(batch_inds + 1) % self.buffer_size, env_indices, :]
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)
            next_fov_locs = self.next_fov_locs[batch_inds, env_indices, :]

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.fov_locs[batch_inds, env_indices, :],
            self.motor_actions[batch_inds, env_indices, :],
            self.sensory_actions[batch_inds, env_indices, :],
            next_obs,
            next_fov_locs,
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )

        reward = data[7]
        discount = np.ones_like(data[7], dtype=np.float32)
        for i in range(1, self.n_step_reward):
            step_reward = self.rewards[(batch_inds+i) % self.buffer_size, env_indices].reshape(-1, 1)
            discount *= np.array([0. if x else self.gamma for x in \
                                  self.dones[(batch_inds+i) % self.buffer_size, env_indices].reshape(-1, 1)], dtype=np.float32).reshape(-1, 1)
            reward += step_reward * discount

        data = (*data[:7], reward, discount)
        
        return NstepRewardDoubleActionWithFovlocReplayBufferSamples(*tuple(map(self.to_torch, data)))
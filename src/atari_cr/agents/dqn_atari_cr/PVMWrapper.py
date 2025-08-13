import gymnasium as gym
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box

from atari_cr.pvm_buffer import PVMBuffer

class PVMWrapper(gym.Wrapper):
    def __init__(self, env, pvm_stack=3, frame_stack=4, mean_pvm=True):
        super().__init__(env)
        obs_shape = env.observation_space.shape  # (C,H,W)
        self.frame_stack = frame_stack
        self.mean_pvm = mean_pvm
        self.pvm = PVMBuffer(pvm_stack, (1, frame_stack, *obs_shape[1:]), mean_pvm=mean_pvm)

        # Beobachtungsraum anpassen
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(frame_stack, *obs_shape[1:]),
            dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.pvm.reset()
        self.pvm.append(np.expand_dims(obs, 0))  # shape (1,C,H,W)
        return self._get_pvm_obs(), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.pvm.append(np.expand_dims(obs, 0))  # shape (1,C,H,W)
        return self._get_pvm_obs(), reward, done, truncated, info

    def _get_pvm_obs(self):
        return self.pvm.get_obs(mode="stack_mean" if self.mean_pvm else "stack_max")[0]



class MultiActionWrapper(gym.Wrapper):
    def __init__(self, env, n_motor: int, n_sensory: int):
        super().__init__(env)
        self.action_space = MultiDiscrete([n_motor, n_sensory])

    def step(self, action):
        # action ist ein Array: [motor_action_id, sensory_action_id]
        action_dict = {
            "motor_action": int(action[0]),
            "sensory_action": self.env.sensory_action_set[int(action[1])]
        }
        return self.env.step(action_dict)


class CRGymWrapper(gym.Wrapper):
    def __init__(self, env, frame_stack=4, pvm_stack=3,
                 sensory_action_quant=(4, 4), mean_pvm=False):
        super().__init__(env)

        # PVM Buffer
        obs_shape = self.observation_space.shape  # (C, H, W)
        self.frame_stack = frame_stack
        self.pvm_stack = pvm_stack
        self.mean_pvm = mean_pvm
        self.obs_shape = obs_shape[1:]  # remove channel
        self.pvm_buffer = PVMBuffer(pvm_stack, (1, frame_stack, *self.obs_shape), mean_pvm=mean_pvm)

        # Create sensory action set (fovea locations)
        self.sensory_action_set = self.create_sensory_action_set(
            self.obs_shape, *sensory_action_quant)

        self.n_motor = self.env.action_space['motor_action'].n  # assuming motor is Discrete(n)
        self.n_sensory = len(self.sensory_action_set)

        # Override action and observation space
        self.action_space = MultiDiscrete([self.n_motor, self.n_sensory], dtype=np.uint16)
        self.observation_space = Box(
            low=0, high=255, shape=(frame_stack, *self.obs_shape), dtype=np.float32)

        # Initial fovea
        self.env.fov_init_loc = self._random_sensory_action()
        self.env.fov_loc = self.env.fov_init_loc

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.pvm_buffer.init_pvm_buffer()
        self.pvm_buffer.append(np.expand_dims(obs, 0))  # shape: (1,C,H,W)
        return self._get_pvm_obs(), info

    def step(self, action):
        motor_action_id = int(np.round(action[0]))
        sensory_action_id = int(np.round(action[1]))
        sensory_action = self.sensory_action_set[sensory_action_id]
        act = {"motor_action": int(motor_action_id), "sensory_action": sensory_action}
        obs, reward, done, truncated, info = self.env.step(act)
        self.pvm_buffer.append(np.expand_dims(obs, 0))
        return self._get_pvm_obs(), reward, done, truncated, info

    def _get_pvm_obs(self):
        return self.pvm_buffer.get_obs(mode="stack_mean" if self.mean_pvm else "stack_max")[0]

    def _random_sensory_action(self):
        return self.sensory_action_set[np.random.choice(len(self.sensory_action_set))]

    @staticmethod
    def create_sensory_action_set(obs_size, x_quantization, y_quantization):
        max_sensory_action_step = np.array(obs_size)
        discrete_coords = [
            np.linspace(0, max_sensory_action_step[i],
                        [x_quantization, y_quantization][i], endpoint=False).astype(int)
            for i in [0, 1]]
        discrete_coords = [coords + int(coords[1] / 2) for coords in discrete_coords]
        return np.stack(np.meshgrid(*discrete_coords)).T.reshape((-1, 2))

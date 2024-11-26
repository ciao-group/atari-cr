import random

import torch
from torch import nn
from torchvision.transforms import Resize
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class QNetwork(nn.Module):
    def __init__(self, env, sensory_action_set):
        super().__init__()

        # Get the size of the different network heads
        assert isinstance(env.single_action_space, spaces.Dict)
        self.motor_action_space_size = env.single_action_space["motor_action"].n
        self.sensory_action_space_size = len(sensory_action_set)

        self.backbone = nn.Sequential( # -> [4,84,84]
            nn.Conv2d(4, 32, 8, stride=4), # -> [32,20,20]
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 4, stride=2), # -> [64,9,9]
            nn.ReLU(),
            # nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=1), # -> [64,7,7]
            nn.ReLU(),
            # nn.BatchNorm2d(64),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )

        self.motor_action_head = nn.Linear(512, self.motor_action_space_size)
        # OPTIONAL: Make sensory_action_head a conv layer with every pixel being a
        # possible sensory action
        self.sensory_action_head = nn.Linear(512, self.sensory_action_space_size)

    def forward(self, x):
        x = self.backbone(x)
        motor_action = self.motor_action_head(x)
        sensory_action = None
        if self.sensory_action_head:
            sensory_action = self.sensory_action_head(x)
        return motor_action, sensory_action

    def chose_action(self, env: gym.vector.VectorEnv, pvm_obs: np.ndarray,
                     epsilon: float, device):
        """
        Epsilon greedy action selection for exploration during training.

        :param ndarray pvm_obs: The Observation
        :param torch.device device: The device to perform computations on
        :param float epsilon: Probability of selecting a random action
        """
        # Execute random motor and sensory action with probability epsilon
        if random.random() < epsilon:
            actions = np.array(
                [env.single_action_space.sample() for _ in range(env.num_envs)])
            motor_actions = np.array([actions[0]["motor_action"]])
            sensory_actions = np.array(
                [random.randint(0, self.sensory_action_space_size-1)])
        else:
            motor_actions, sensory_actions = self.chose_eval_action(pvm_obs, device)

        return motor_actions, sensory_actions

    def chose_eval_action(self, pvm_obs: np.ndarray, device: torch.device):
        """
        Greedy action selection exploiting the best known policy.

        :param ndarray pvm_obs: The Observation
        :param torch.device device: The device to perform computations on
        """
        resize = Resize(pvm_obs.shape[2:])
        motor_q_values, sensory_q_values = self(resize(
            torch.from_numpy(pvm_obs)).to(device))
        motor_actions = torch.argmax(motor_q_values, dim=1).cpu().numpy()
        sensory_actions = torch.argmax(sensory_q_values, dim=1).cpu().numpy()

        return motor_actions, sensory_actions

# @torch.compile()
class SelfPredictionNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        assert isinstance(env.single_action_space, spaces.Dict), \
            "SelfPredictionNetwork only works with Dict action space"
        motor_action_space_size = env.single_action_space["motor_action"].n

        self.backbone = nn.Sequential(
            nn.Conv2d(8, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(512, motor_action_space_size),
        )

        self.loss = nn.CrossEntropyLoss()

    def get_loss(self, x, target) -> torch.Tensor:
        return self.loss(x, target)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

import random
from typing import Optional

import torch
from torch import nn
from torchvision.transforms import Resize
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class QNetwork(nn.Module):
    def __init__(self, env, sensory_action_set, pause_feat: bool, s_action_feat: bool):
        super().__init__()
        self.pause_feat = pause_feat
        self.s_action_feat = s_action_feat

        # Get the size of the different network heads
        assert isinstance(env.single_action_space, spaces.Dict)
        self.motor_action_space_size = env.single_action_space["motor_action"].n
        self.sensory_action_space_size = len(sensory_action_set)

        self.conv_backbone = nn.Sequential( # -> [4,84,84]
            nn.Conv2d(4, 32, 8, stride=4), # -> [32,20,20]
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 4, stride=2), # -> [64,9,9]
            nn.ReLU(),
            # nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=1), # -> [64,7,7]
            nn.ReLU(),
            # nn.BatchNorm2d(64),
            nn.Flatten(), # -> [3136]
        )

        # Here is where additional features are given to the model
        dim = 3136
        if pause_feat: dim += 1
        if s_action_feat: dim += 2

        self.linear_backbone = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
        )

        self.motor_action_head = nn.Linear(512, self.motor_action_space_size)
        self.sensory_action_head = nn.Linear(512, self.sensory_action_space_size)

    def forward(self, x, consecutive_pauses: Optional[torch.Tensor] = None,
                s_actions: Optional[torch.Tensor] = None):
        """
        :param Tensor[B,4,84,84] x:
        :param Tensor[B] consecutive_pauses:
        :param Tensor[B,2] s_action:
        """
        x = self.conv_backbone(x) # -> [B,3136]
        if self.pause_feat:
            x = torch.cat(
                [x, consecutive_pauses.to(torch.float32).to(x.device).unsqueeze(-1)],
                dim=1)
        if self.s_action_feat:
            x = torch.cat(
                [x, s_actions.to(x.device).view([len(x),-1])],
                dim=1)
        x = self.linear_backbone(x)

        return self.motor_action_head(x), self.sensory_action_head(x)

    def chose_action(self, env: gym.vector.VectorEnv, pvm_obs: np.ndarray,
            epsilon: float, device, consecutive_pauses: Optional[np.ndarray] = None,
            prev_pause_actions: Optional[torch.Tensor] = None):
        """
        Epsilon greedy action selection for exploration during training.

        :param Array[N,4,84,84] pvm_obs: PVM Obs for every env in the vec env
        :param torch.device device: The device to perform computations on
        :param float epsilon: Probability of selecting a random action
        :param Tensor[N;i64] consecutive_pauses: Pause count for every env
        """
        # Execute random motor and sensory action with probability epsilon
        if random.random() < epsilon:
            actions = np.array(
                [env.single_action_space.sample() for _ in range(env.num_envs)])
            motor_actions = np.array([actions[0]["motor_action"]])
            sensory_actions = np.array(
                [random.randint(0, self.sensory_action_space_size-1)])
        else:
            motor_actions, sensory_actions = self.chose_eval_action(
                pvm_obs, device, consecutive_pauses, prev_pause_actions)

        return motor_actions, sensory_actions

    def chose_eval_action(self, pvm_obs: np.ndarray, device: torch.device,
            consecutive_pauses: Optional[torch.Tensor] = None,
            prev_pause_actions: Optional[torch.Tensor] = None):
        """
        Greedy action selection exploiting the best known policy.

        :param Array[N,4,84,84] pvm_obs: PVM Obs for every env in the vec env
        :param torch.device device: The device to perform computations on
        :param Tensor[N;i64] consecutive_pauses: Pause count for every env
        """
        resize = Resize(pvm_obs.shape[2:])
        motor_q_values, sensory_q_values = self(resize(
            torch.from_numpy(pvm_obs)).to(device), consecutive_pauses, prev_pause_actions)
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

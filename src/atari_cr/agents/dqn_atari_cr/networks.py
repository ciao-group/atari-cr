import random
from typing import Optional

import torch
from torch import nn
from torchvision.transforms import Resize
import numpy as np
from gymnasium import spaces
from gymnasium.vector import VectorEnv

class QNetwork(nn.Module):
    def __init__(self, env, sensory_out_dim: int, pause_feat: bool,
                 s_action_feat: bool):
        super().__init__()
        self.pause_feat = pause_feat
        self.s_action_feat = s_action_feat

        # Get the size of the different network heads
        assert isinstance(env.single_action_space, spaces.Dict)
        self.motor_out_dim = env.single_action_space["motor_action"].n
        self.sensory_out_dim = sensory_out_dim

        self.conv_backbone = nn.Sequential( # -> [4,84,84]
            nn.Conv2d(4, 32, 8, stride=4), # -> [32,20,20]
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), # -> [64,9,9]
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), # -> [64,7,7]
            # nn.BatchNorm2d(64),
            nn.ReLU(),
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

        self.motor_action_head = nn.Linear(512, self.motor_out_dim)
        self.sensory_action_head = nn.Linear(512, self.sensory_out_dim)

    def forward(self, x, consecutive_pauses: Optional[torch.Tensor] = None,
                s_actions: Optional[torch.Tensor] = None):
        """
        :param Tensor[B,4,84,84;f32] x: Stack of greyscale frames with values between
            0 and 1
        :param Tensor[B] consecutive_pauses:
        :param Tensor[B,2] s_action: Coordinates of the previous sensory action.
            Normalized between 0 and 1
        """
        # assert consecutive_pauses.max() <= 1 and s_actions.max() <= 1
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

    def chose_action(self, pvm_obs: np.ndarray, device,
            env: VectorEnv, sensory_noops: np.ndarray,
            consecutive_pauses: Optional[np.ndarray] = None,
            prev_pause_actions: Optional[torch.Tensor] = None, epsilon = 0.):
        """
        Epsilon greedy action selection for exploration during training.

        :param Array[B,4,84,84] pvm_obs: PVM Obs for every env in the vec env
        :param torch.device device: The device to perform computations on
        :param float epsilon: Probability of selecting a random action
        :param Tensor[B;i64] consecutive_pauses: Pause count for every env
        :param Tensor[B,2] prev_pause_actions: Coordinates of the previous sensory
            action. Normalized between 0 and 1
        :param Array[B,2] masked_actions: IDs of motor and sensory actions to be
            masked out
        :param Array[B] pause_ids: ID of the pause action in every env
        """
        b = pvm_obs.shape[0] # Batch size, number of envs in VectorEnv
        # Execute random motor and sensory action with probability epsilon
        if random.random() < epsilon:
            motor_q_values = np.random.dirichlet(np.ones(self.motor_out_dim), b)
            sensory_q_values = np.random.dirichlet(np.ones(self.sensory_out_dim), b)
        else:
            resize = Resize(pvm_obs.shape[2:])
            motor_q_values, sensory_q_values = self(
                resize(torch.from_numpy(pvm_obs)).to(device),
                consecutive_pauses, prev_pause_actions)
            motor_q_values = motor_q_values.detach().cpu().numpy()
            sensory_q_values = sensory_q_values.detach().cpu().numpy()

        # Mask out too many pauses in a row
        mask = np.where(
            [e.consecutive_pauses >= e.consecutive_pause_limit for e in env.envs])[0]
        motor_q_values[mask,-1] = -np.inf

        motor_actions = np.argmax(motor_q_values, axis=1)

        # Mask out pauses without fovea movement
        motor_mask = np.where(motor_actions == env.envs[0].pause_action)[0]
        sensory_q_values[motor_mask,sensory_noops[motor_mask]] = -np.inf

        sensory_actions = np.argmax(sensory_q_values, axis=1)

        return motor_actions, sensory_actions

# @torch.compile()
class SelfPredictionNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        assert isinstance(env.single_action_space, spaces.Dict), \
            "SelfPredictionNetwork only works with Dict action space"
        motor_out_dim = env.single_action_space["motor_action"].n

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
            nn.Linear(512, motor_out_dim),
        )

        self.loss = nn.CrossEntropyLoss()

    def get_loss(self, x, target) -> torch.Tensor:
        return self.loss(x, target)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

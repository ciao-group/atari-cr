from functools import partial
from typing import Callable, Optional, Tuple, Union, Dict
import numpy as np
from torch import nn
import torch
from torchvision.models.vision_transformer import VisionTransformer
from torchvision.transforms import Resize
from tqdm import tqdm
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box, Discrete, Dict
import stable_baselines3.common.preprocessing as sb_preprocessing

from active_gym import AtariEnvArgs
from active_gym.atari_env import AtariEnv
from atari_cr.atari_head.utils import VISUAL_DEGREE_SCREEN_SIZE
from atari_cr.models import EpisodeRecord, FovType

class tqdm(tqdm):
    @property
    def format_dict(self):
        d = super().format_dict

        # Make the bar yellow
        d.update({"colour": "yellow"})

        # ... and green when finished
        if d["n"] == d["total"]: d.update({"colour": "green"})

        return d

class ViTEmbedder(VisionTransformer):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        device="cpu"
    ):
        nn.Module.__init__(self)
        torch._assert(
            image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer
        self.device = device

        self.conv_proj = nn.Conv2d(
            in_channels=3, out_channels=hidden_dim, kernel_size=patch_size,
            stride=patch_size
        )

        self.seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.seq_length += 1

    def forward(self, x: torch.Tensor):
        """ The ViT forward pass, cut before ViTs own encoder """
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1).to(self.device)
        x = torch.cat([batch_class_token, x], dim=1)

        return x

# Taken from stable_baselines3.common.preprocessing
def get_action_dim(action_space: spaces.Space) -> Union[int, Dict]:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Dict):
        action_dims = []
        for key in action_space:
            action_dims.append(get_action_dim(action_space[key]))
        return action_dims
    else:
        return sb_preprocessing.get_action_dim(action_space)

class FixedFovealEnv(gym.Wrapper):
    def __init__(self, env: AtariEnv, args: AtariEnvArgs, pause_cost = 0.01,
            saccade_cost_scale = 0.001, fov: FovType = "window", no_pauses = False,
            consecutive_pause_limit = 20):
        super().__init__(env)
        self.fov_size: Tuple[int, int] = args.fov_size
        self.fov: FovType = fov
        self.fov_init_loc: Tuple[int, int] = args.fov_init_loc
        assert (np.array(self.fov_size) <
                np.array(env.obs_size)).all()
        self.no_pauses = no_pauses

        # Get sensory action space for the sensory action mode
        self.relative_sensory_actions = args.sensory_action_mode == "relative"
        if self.relative_sensory_actions:
            self.sensory_action_space = np.array(args.sensory_action_space)
        elif self.fov in ["gaussian", "exponential"]:
            self.sensory_action_space = np.array(env.obs_size)
        elif self.fov == "window":
            self.sensory_action_space = \
                np.array(env.obs_size) - np.array(self.fov_size)

        self.resize: Resize = Resize(env.obs_size) \
            if args.resize_to_full else None

        self.action_space = Dict({
            # One additional action lets the agent stop the game to perform a
            # sensory action without the game progressing
            "motor_action": Discrete(
                env.action_space.n if no_pauses else env.action_space.n + 1),
            "sensory_action": Box(low=self.sensory_action_space[0],
                                 high=self.sensory_action_space[1], dtype=int),
        })
        self.pause_action = None if no_pauses else \
            self.action_space["motor_action"].n - 1

        # How much the agent is punished for large eye movements
        self.saccade_cost_scale = saccade_cost_scale
        self.visual_degrees_per_pixel = np.array(VISUAL_DEGREE_SCREEN_SIZE) / \
            np.array(self.get_wrapper_attr("obs_size"))

        # Whether the previous action was a pause action
        self.prev_pause_action = 0
        # Count and log the number of pauses made and their cost
        self.pause_cost = pause_cost
        self.consecutive_pause_limit = consecutive_pause_limit

        # Attributes for recording episodes
        self.record = args.record
        self.prev_episode: Optional[EpisodeRecord] = None
        self.env: AtariEnv

    def _init_fov_loc(self):
        self.fov_loc = np.rint(np.array(self.fov_init_loc, copy=True)).astype(np.int32)

    def reset_record_buffer(self):
        self.env.record_buffer["fov_size"] = self.fov_size
        self.env.record_buffer["fov_loc"] = []

    def reset(self):
        full_state, info = self.env.reset()
        self._init_fov_loc()
        fov_state = self._get_fov_state(full_state)
        info["fov_loc"] = self.fov_loc.copy()
        if self.env.record:
            self.reset_record_buffer()
            self.save_transition(info["fov_loc"])
        return fov_state, info

    def _clip_to_valid_fov(self, loc):
        return np.rint(np.clip(
            loc, 0, np.array(self.env.obs_size) - np.array(self.fov_size))).astype(int)

    def _clip_to_valid_sensory_action_space(self, action):
        return np.rint(np.clip(action, *self.sensory_action_space)).astype(int)

    def _get_fov_state(self, full_state):
        fov_state = full_state[..., self.fov_loc[0]:self.fov_loc[0]+self.fov_size[0],
                                    self.fov_loc[1]:self.fov_loc[1]+self.fov_size[1]]

        mask = np.zeros_like(full_state)
        mask[..., self.fov_loc[0]:self.fov_loc[0]+self.fov_size[0],
                self.fov_loc[1]:self.fov_loc[1]+self.fov_size[1]] = fov_state
        fov_state = mask

        return fov_state

    def _fov_step(self, full_state, action):
        if type(action) is torch.Tensor:
            action = action.detach().cpu().numpy()
        elif type(action) is Tuple:
            action = np.array(action)

        action = self._clip_to_valid_fov(action)
        self.fov_loc = action

        fov_state = self._get_fov_state(full_state)

        return fov_state

    def save_transition(self, fov_loc):
        # print ("saving one transition")
        self.env.record_buffer["fov_loc"].append(fov_loc)

    def step(self, action):
        """
        action : {"motor_action":
                  "sensory_action": }
        """
        state, reward, done, truncated, info = \
            self.env.step(action=action["motor_action"])
        fov_state = self._fov_step(full_state=state, action=action["sensory_action"])
        info["fov_loc"] = self.fov_loc.copy()
        if self.record:
            if not done:
                self.save_transition(info["fov_loc"])
        return fov_state, reward, done, truncated, info

    @property
    def unwrapped(self):
        """
        Grabs unwrapped environment

        Returns:
            env (MujocoEnv): Unwrapped environment
        """
        if hasattr(self.env, "unwrapped"):
            return self.env.unwrapped
        else:
            return self.env

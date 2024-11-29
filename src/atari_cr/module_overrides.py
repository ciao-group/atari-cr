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
from gymnasium.spaces import Box

import stable_baselines3.common.preprocessing as sb_preprocessing

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
    def __init__(self, env: gym.Env, args):
        super().__init__(env)
        self.fov_size: Tuple[int, int] = args.fov_size
        self.fov_init_loc: Tuple[int, int] = args.fov_init_loc
        assert (np.array(self.fov_size) < np.array(self.obs_size)).all()

        self.sensory_action_mode: str = args.sensory_action_mode # "absolute","relative"
        if self.sensory_action_mode == "relative":
            self.sensory_action_space = np.array(args.sensory_action_space)
        elif self.sensory_action_mode == "absolute":
            self.sensory_action_space = np.array(self.obs_size) \
                                        - np.array(self.fov_size)

        self.resize: Resize = Resize(self.env.obs_size) if args.resize_to_full else None

        self.mask_out: bool = args.mask_out

        # set gym.Env attribute
        self.action_space = Dict({
            "motor_action": self.env.action_space,
            "sensory_action": Box(low=self.sensory_action_space[0],
                                 high=self.sensory_action_space[1], dtype=int),
        })

        if args.mask_out:
            self.observation_space = Box(low=-1., high=1.,
                                         shape=(self.env.frame_stack,)+self.env.obs_size,
                                         dtype=np.float32)
        elif args.resize_to_full:
            self.observation_space = Box(low=-1., high=1.,
                                         shape=(self.env.frame_stack,)+self.env.obs_size,
                                         dtype=np.float32)
        else:
            self.observation_space = Box(low=-1., high=1.,
                                         shape=(self.env.frame_stack,)+self.fov_size,
                                         dtype=np.float32)

        # init fov location
        # The location of the upper left corner of the fov image,
        # on the original observation plane
        self.fov_loc: np.ndarray = np.empty_like(self.fov_init_loc)  #
        self._init_fov_loc()

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

        if self.mask_out:
            mask = np.zeros_like(full_state)
            mask[..., self.fov_loc[0]:self.fov_loc[0]+self.fov_size[0],
                    self.fov_loc[1]:self.fov_loc[1]+self.fov_size[1]] = fov_state
            fov_state = mask
        elif self.resize:
            fov_state = self.resize(torch.from_numpy(fov_state))
            fov_state = fov_state.numpy()

        return fov_state

    def _fov_step(self, full_state, action):
        if type(action) is torch.Tensor:
            action = action.detach().cpu().numpy()
        elif type(action) is Tuple:
            action = np.array(action)

        if self.sensory_action_mode == "absolute":
            action = self._clip_to_valid_fov(action)
            self.fov_loc = action
        elif self.sensory_action_mode == "relative":
            action = self._clip_to_valid_sensory_action_space(action)
            fov_loc = self.fov_loc + action
            self.fov_loc = self._clip_to_valid_fov(fov_loc)

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
        # print ("in env", action, action["motor_action"], action["sensory_action"])
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

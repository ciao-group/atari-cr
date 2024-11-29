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
from atari_cr.atari_head.dataset import SCREEN_SIZE
from atari_cr.atari_head.utils import VISUAL_DEGREE_SCREEN_SIZE
from atari_cr.graphs.eccentricity import A, B, C
from atari_cr.models import EpisodeInfo, EpisodeRecord, FovType, StepInfo
from atari_cr.pauseable_env import MASKED_ACTION_PENTALTY
from atari_cr.utils import EMMA_fixation_time

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

    def reset(self):
        self.state, info = self.env.reset() # -> [4,84,84;f64]
        assert info == {'raw_reward': 0}

        self.timestep = -1
        self.frames = []
        # self.obs appended to from outside the environment
        self.obs = []
        self.step_infos = []
        self.episode_info = EpisodeInfo.new()
        self.consecutive_pauses = 0

        self.fov_loc = np.array(self.fov_init_loc, copy=True).astype(np.int32)
        fov_obs = self._crop_observation(self.state)

        return fov_obs, info

    def step(self, action):
        # Log the state before the action is taken
        self.timestep += 1
        step_info = StepInfo.new()
        step_info["timestep"] = self.timestep
        step_info["fov_loc"] = self.fov_loc.tolist()
        step_info["motor_action"] = action["motor_action"]
        step_info["sensory_action"] = action["sensory_action"]
        step_info["pauses"] = int(action["motor_action"] == self.pause_action)
        self.frames.append(self.unwrapped.render())

        # Save the previous fov_loc to calculate the saccade cost
        prev_fov_loc = self.fov_loc.copy()

        # Actual pause step
        if step_info["pauses"]:
            done, truncated = False, False
            info = {"raw_reward": 0}

            # Mask out unwanted pause behavior
            if self.consecutive_pauses > self.consecutive_pause_limit:
                # Too many pauses in a row
                reward = MASKED_ACTION_PENTALTY
                step_info["prevented_pauses"] = 1
                truncated = True
            elif np.all(self.fov_loc == action["sensory_action"]):
                # No action pause
                reward = MASKED_ACTION_PENTALTY
                step_info["no_action_pauses"] = 1
            else:
                # Normal pause
                reward = -self.pause_cost

            # Log another pause
            if self.prev_pause_action:
                self.consecutive_pauses += 1
            else:
                self.consecutive_pauses = 0
            self.prev_pause_action = step_info["pauses"]

        else:
            self.consecutive_pauses = 0
            # Normal step, the state is saved for the next pause step
            self.state, reward, done, truncated, info = \
                self.env.step(action=action["motor_action"])

        # Sensory step
        fov_state = self._fov_step(
            full_state=self.state, action=action["sensory_action"])

        # Add costs for the time it took the agent to move its fovea
        if self.saccade_cost_scale:
            visual_degree_distance = np.sqrt(np.sum(np.square(
                (self.fov_loc - prev_fov_loc) * self.visual_degrees_per_pixel) ))
            _, total_emma_time, fov_moved = EMMA_fixation_time(visual_degree_distance)
            step_info["emma_time"] = total_emma_time
            step_info["saccade_cost"] = self.saccade_cost_scale * total_emma_time
            reward -= step_info["saccade_cost"]

        # Log the results of taking an action
        step_info["raw_reward"] = info["raw_reward"]
        step_info["done"] = done
        step_info["truncated"] = truncated
        # Episode info stores cumulative sums of the step info keys
        self.episode_info: EpisodeInfo
        for key in self.episode_info.keys():
            self.episode_info[key] += step_info[key]
        step_info["episode_info"] = self.episode_info.copy()
        step_info["consecutive_pauses"] = self.consecutive_pauses
        self.step_infos.append(step_info)

        # Log the infos of all steps in one record for the
        # entire episode
        if done or truncated:
            self.prev_episode = EpisodeRecord(
                np.stack(self.frames),
                EpisodeRecord.annotations_from_step_infos(self.step_infos),
                { "fov_size": self.fov_size, "fov": self.fov },
                np.stack(self.obs) if self.obs else None
            )

        return fov_state, reward, done, truncated, step_info

    def add_obs(self, obs: np.ndarray):
        """ :param Array[4,84,84] obs: Frame stack, only last frame is saved """
        self.obs.append(obs[-1])

    def _clip_to_valid_fov(self, loc: np.ndarray):
        """ :param Array[W,H] loc: """
        return np.clip(loc, [0,0], self.sensory_action_space).astype(int)

    def _fov_step(self, full_state, action):
        """
        Changes self.fov_loc by the given action and returns a version of the full
        state that is cropped to where the new self.fov_loc is

        :param Array[2] action:
        :returns Array[4,84,84]:
        """
        # Move the fovea
        if self.relative_sensory_actions:
            action = self._clip_to_valid_fov(action)
            action = self.fov_loc + action
        self.fov_loc = self._clip_to_valid_fov(action)

        fov_state = self._crop_observation(full_state)

        return fov_state

    def _crop_observation(self, full_state: np.ndarray):
        """
        Get a version of the full_state that is cropped to the fovea around fov_loc if
        fov is set to 'window'. Otherwise get a mask over the output

        :param Array[4,84,84] full_state: Stack of four greyscale images of type float64
        """
        masked_state = np.zeros_like(full_state)
        if self.fov == "gaussian":
            distances_from_fov = self._pixel_eccentricities(
                full_state.shape[-2:], self.fov_loc)
            sigma = self.fov_size[0] / 2
            gaussian = np.exp(-np.sum(
                np.square(distances_from_fov) / (2 * np.square(sigma)),
                axis=0)) # -> [84,84]
            gaussian /= gaussian.max()
            masked_state = full_state * gaussian
        elif self.fov == "window":
            crop = full_state[...,
                self.fov_loc[0]:self.fov_loc[0] + self.fov_size[0],
                self.fov_loc[1]:self.fov_loc[1] + self.fov_size[1]
            ]
            # Fill the area outside the crop with zeros
            masked_state[...,
                self.fov_loc[0]:self.fov_loc[0] + self.fov_size[0],
                self.fov_loc[1]:self.fov_loc[1] + self.fov_size[1]
            ] = crop
        elif self.fov == "exponential":
            pixel_eccentricities = self._pixel_eccentricities(
                full_state.shape[-2:], self.fov_loc)
            # Convert from pixels to visual degrees
            eccentricities = (pixel_eccentricities.transpose([1,2,0]) *\
                (np.array(VISUAL_DEGREE_SCREEN_SIZE) / np.array(SCREEN_SIZE))
                ).transpose(2,0,1)
            # Absolute 1D distances
            abs_distances = np.sqrt(np.square(eccentricities).sum(axis=0))

            mask = A * np.exp(B * abs_distances) + C
            mask /= mask.max()
            masked_state = full_state * mask

        return masked_state


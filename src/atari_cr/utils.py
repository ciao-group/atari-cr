"""
Borrow from stable-baselines3
Due to dependencies incompability, we cherry-pick codes here
"""
import os
import random
import re
from datetime import datetime
import warnings
from typing import Any, Dict, List, Tuple, Union

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from PIL import Image

from gymnasium import spaces

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def is_image_space_channels_first(observation_space: spaces.Box) -> bool:
    """
    Check if an image observation space (see ``is_image_space``)
    is channels-first (CxHxW, True) or channels-last (HxWxC, False).
    Use a heuristic that channel dimension is the smallest of the three.
    If second dimension is smallest, raise an exception (no support).
    :param observation_space:
    :return: True if observation space is channels-first image, False if channels-last.
    """
    smallest_dimension = np.argmin(observation_space.shape).item()
    if smallest_dimension == 1:
        warnings.warn("Treating image space as channels-last, while second dimension was smallest of the three.")
    return smallest_dimension == 0


def is_image_space(
    observation_space: spaces.Space,
    check_channels: bool = False,
    normalized_image: bool = False,
) -> bool:
    """
    Check if a observation space has the shape, limits and dtype
    of a valid image.
    The check is conservative, so that it returns False if there is a doubt.
    Valid images: RGB, RGBD, GrayScale with values in [0, 255]
    :param observation_space:
    :param check_channels: Whether to do or not the check for the number of channels.
        e.g., with frame-stacking, the observation space may have more channels than expected.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    :return:
    """
    check_dtype = check_bounds = not normalized_image
    if isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 3:
        # Check the type
        if check_dtype and observation_space.dtype != np.uint8:
            return False

        # Check the value range
        incorrect_bounds = np.any(observation_space.low != 0) or np.any(observation_space.high != 255)
        if check_bounds and incorrect_bounds:
            return False

        # Skip channels check
        if not check_channels:
            return True
        # Check the number of channels
        if is_image_space_channels_first(observation_space):
            n_channels = observation_space.shape[0]
        else:
            n_channels = observation_space.shape[-1]
        # GrayScale, RGB, RGBD
        return n_channels in [1, 3, 4]
    return False



def preprocess_obs(
    obs: torch.Tensor,
    observation_space: spaces.Space,
    normalize_images: bool = True,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Preprocess observation to be to a neural network.
    For images, it normalizes the values by dividing them by 255 (to have values in [0, 1])
    For discrete observations, it create a one hot vector.
    :param obs: Observation
    :param observation_space:
    :param normalize_images: Whether to normalize images or not
        (True by default)
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        if normalize_images and is_image_space(observation_space):
            return obs.float() / 255.0
        return obs.float()

    elif isinstance(observation_space, spaces.Discrete):
        # One hot encoding and convert to float to avoid errors
        return F.one_hot(obs.long(), num_classes=observation_space.n).float()

    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Tensor concatenation of one hot encodings of each Categorical sub-space
        return torch.cat(
            [
                F.one_hot(obs_.long(), num_classes=int(observation_space.nvec[idx])).float()
                for idx, obs_ in enumerate(torch.split(obs.long(), 1, dim=1))
            ],
            dim=-1,
        ).view(obs.shape[0], sum(observation_space.nvec))

    elif isinstance(observation_space, spaces.MultiBinary):
        return obs.float()

    elif isinstance(observation_space, spaces.Dict):
        # Do not modify by reference the original observation
        assert isinstance(obs, Dict), f"Expected dict, got {type(obs)}"
        preprocessed_obs = {}
        for key, _obs in obs.items():
            preprocessed_obs[key] = preprocess_obs(_obs, observation_space[key], normalize_images=normalize_images)
        return preprocessed_obs

    else:
        raise NotImplementedError(f"Preprocessing not implemented for {observation_space}")


def get_obs_shape(
    observation_space: spaces.Space,
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """
    Get the shape of the observation (useful for the buffers).
    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        if type(observation_space.n) in [tuple, list, np.ndarray]:
            return tuple(observation_space.n)
        else:
            return (int(observation_space.n),)
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


def get_flattened_obs_dim(observation_space: spaces.Space) -> int:
    """
    Get the dimension of the observation space when flattened.
    It does not apply to image observation space.
    Used by the ``FlattenExtractor`` to compute the input shape.
    :param observation_space:
    :return:
    """
    # See issue https://github.com/openai/gym/issues/1915
    # it may be a problem for Dict/Tuple spaces too...
    if isinstance(observation_space, spaces.MultiDiscrete):
        return sum(observation_space.nvec)
    else:
        # Use Gym internal method
        return spaces.utils.flatdim(observation_space)

def check_for_nested_spaces(obs_space: spaces.Space):
    """
    Make sure the observation space does not have nested spaces (Dicts/Tuples inside Dicts/Tuples).
    If so, raise an Exception informing that there is no support for this.
    :param obs_space: an observation space
    :return:
    """
    if isinstance(obs_space, (spaces.Dict, spaces.Tuple)):
        sub_spaces = obs_space.spaces.values() if isinstance(obs_space, spaces.Dict) else obs_space.spaces
        for sub_space in sub_spaces:
            if isinstance(sub_space, (spaces.Dict, spaces.Tuple)):
                raise NotImplementedError(
                    "Nested observation spaces are not supported (Tuple/Dict space inside Tuple/Dict space)."
                )


def get_device(device: Union[torch.device, str] = "auto") -> torch.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.
    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to torch.device
    device = torch.device(device)

    # Cuda not available
    if device.type == torch.device("cuda").type and not torch.cuda.is_available():
        return torch.device("cpu")

    return device


def get_timestr() -> str:
    current_datetime = datetime.now()
    return current_datetime.strftime("%m-%d-%H-%M-%S")


def get_spatial_emb_indices(loc: np.ndarray,
                            full_img_size=(4, 84, 84),
                            img_size=(4, 21, 21),
                            patch_size=(7, 7)) -> np.ndarray:
    # loc (2,)
    _, H, W = full_img_size
    _, h, w = img_size
    p1, p2 = patch_size

    st_x = loc[0] // p1
    st_y = loc[1] // p2

    ed_x = (loc[0] + h) // p1
    ed_y = (loc[1] + w) // p2

    ix, iy = np.meshgrid(np.arange(st_x, ed_x, dtype=np.int64),
                            np.arange(st_y, ed_y, dtype=np.int64), indexing="ij")

    # print (ix, iy)
    indicies = (ix * H // p1 + iy).reshape(-1)

    return indicies

def get_spatial_emb_mask(loc,
                         mask,
                         full_img_size=(4, 84, 84),
                         img_size=(4, 21, 21),
                         patch_size=(7, 7),
                         latent_dim=144) -> np.ndarray:
    B, T, _ = loc.size()
    # return torch.randn_like()
    loc = loc.reshape(-1, 2)
    _, H, W = full_img_size
    _, h, w = img_size
    p1, p2 = patch_size
    num_tokens = h*w//p1//p2
    # print ("num_tokens", num_tokens)

    st_x = loc[..., 0] // p1
    st_y = loc[..., 1] // p2

    ed_x = (loc[..., 0] + h) // p1
    ed_y = (loc[..., 1] + w) // p2

    # mask = np.zeros(((32*6, H//p1, W//p2, latent_dim)), dtype=np.bool_)
    # mask = torch.zeros((32*6, H//p1, W//p2, latent_dim), dtype=torch.bool)
    mask[:] = False
    for i in range(B*T):
        # print (self.spatial_emb[0, st_x[i]:ed_x[i], st_y[i]:ed_y[i]].size())
        mask[i, st_x[i]:ed_x[i], st_y[i]:ed_y[i]] = True
    return mask[:B*T]

def weight_init_drq(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def schedule_drq(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)

def get_sugarl_reward_scale_robosuite(task_name) -> float:
    if task_name == "Lift":
        sugarl_reward_scale = 150/500
    elif task_name == "ToolHang":
        sugarl_reward_scale = 100/500
    else:
        sugarl_reward_scale = 100/500
    return sugarl_reward_scale


def get_sugarl_reward_scale_dmc(domain_name, task_name) -> float:
    if domain_name == "ball_in_cup" and task_name == "catch":
        sugarl_reward_scale = 320/500
    elif domain_name == "cartpole" and task_name == "swingup":
        sugarl_reward_scale = 380/500
    elif domain_name == "cheetah" and task_name == "run":
        sugarl_reward_scale = 245/500
    elif domain_name == "dog" and task_name == "fetch":
        sugarl_reward_scale = 4.5/500
    elif domain_name == "finger" and task_name == "spin":
        sugarl_reward_scale = 290/500
    elif domain_name == "fish" and task_name == "swim":
        sugarl_reward_scale = 64/500
    elif domain_name == "reacher" and task_name == "easy":
        sugarl_reward_scale = 200/500
    elif domain_name == "walker" and task_name == "walk":
        sugarl_reward_scale = 290/500
    else:
        return 1.

    return sugarl_reward_scale

def get_sugarl_reward_scale_atari(game) -> float:
    base_scale = 4.0
    sugarl_reward_scale = 1/200
    if game in ["alien", "assault", "asterix", "battle_zone", "seaquest", "qbert", "private_eye", "road_runner"]:
        sugarl_reward_scale = 1/100
    elif game in ["kangaroo", "krull", "chopper_command", "demon_attack"]:
        sugarl_reward_scale = 1/200
    elif game in ["up_n_down", "frostbite", "ms_pacman", "amidar", "gopher", "boxing"]:
        sugarl_reward_scale = 1/50
    elif game in ["hero", "jamesbond", "kung_fu_master"]:
        sugarl_reward_scale = 1/25
    elif game in ["crazy_climber"]:
        sugarl_reward_scale = 1/20
    elif game in ["freeway"]:
        sugarl_reward_scale = 1/1600
    elif game in ["pong"]:
        sugarl_reward_scale = 1/800
    elif game in ["bank_heist"]:
        sugarl_reward_scale = 1/250
    elif game in ["breakout"]:
        sugarl_reward_scale = 1/35
    sugarl_reward_scale = sugarl_reward_scale * base_scale
    return sugarl_reward_scale

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def grid_image2(images: Union[np.ndarray, torch.Tensor]):
    """
    Save a grid of rgb images seperated by colored lines under 'debug.png'

    :param Array[n_rows, n_cols, width, height, 3] array: Structured array of images
    """
    assert len(images.shape) in [4, 5], "Only works for images of shape [n_rows, n_cols, x, y, n_channels] or [n_rows, n_cols, x, y]"

    # Convert torch tensor to numpy array
    if isinstance(images, torch.Tensor): images = images.numpy()

    # Convert to float32
    if isinstance(images, np.uint8): images = images.astype(np.float32) / 256

    # Set grid size (e.g., 3x3 grid)
    grid_size = images.shape[:2]

    fig, axes = plt.subplots(*grid_size, figsize=(16, 16))
    axes = axes.flatten()
    images = images.reshape([-1, *images.shape[2:]])

    for img, ax in zip(images, axes):
        ax.imshow(img, cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("debug.png")

def grid_image(array: Union[np.ndarray, torch.Tensor], line_color=[255, 0, 0], line_width=1):
    """
    Display a grid of rgb images seperated by colored lines

    :param Array[n_rows, n_cols, width, height, 3] array: Structured array of images
    """
    assert len(array.shape) in [4, 5], "Only works for array of shape [n_rows, n_cols, x, y, n_channels] or [n_rows, n_cols, x, y]"

    # Convert torch tensor to numpy array
    if isinstance(array, torch.Tensor): array = array.numpy()

    # Convert greyscale to RGB
    if len(array.shape) == 4:
        array = np.broadcast_to(array[:,:,:,:,np.newaxis], [*array.shape, 3])
    if array.shape[-1] == 1:
        array = np.broadcast_to(array, [*array.shape[:-1], 3])
    n_rows, n_cols, y, x, n_channels = array.shape

    # Convert to uint8
    if not array.flags["WRITEABLE"]: array = array.copy()
    if array.dtype == np.float32:
        for i in range(len(array)):
            for j in range(len(array[i])):
                if array[i,j].min() == array[i,j].max():
                    # Clip arrays with all elements having the same value to be between 0 and 1
                    array[i,j] = np.max([array[i,j], np.zeros(array[i,j].shape)])
                    array[i,j] = np.min([array[i,j], np.ones(array[i,j].shape)])
                else:
                    # Or normalize to be between 0 and 1
                    array[i,j] = array[i,j] - array[i,j].min()
                    array[i,j] = array[i,j] / array[i,j].max()
                # Scale to be between 0 and 255
                array[i,j] *= 255
        array = array.astype(np.uint8)

    # Create a new RGB array to hold the grid with separating lines
    grid_size = (y * n_rows + line_width * (n_rows - 1), x * n_cols + line_width * (n_cols - 1), n_channels)
    grid = np.zeros(grid_size, dtype=np.uint8)

    # Create colored lines
    for i in range(1, n_rows):
        grid[i*(y+line_width)-line_width:i*(y+line_width), :] = line_color
    for j in range(1, n_cols):
        grid[:, j*(x+line_width)-line_width:j*(x+line_width)] = line_color

    # Plot each image in the grid
    for i in range(n_rows):
        for j in range(n_cols):
            y_start = i * (y + line_width)
            x_start = j * (x + line_width)
            grid[y_start:y_start+y, x_start:x_start+x] = array[i, j]

    return grid

def debug_array(array: Union[np.ndarray, torch.Tensor, List[torch.Tensor]]):
    """
    Saves a 2D, 3D or 4D greyscale array as an image under 'debug.png'.
    """
    # Turn a list of arrays or tensors into one array
    if isinstance(array, List):
        for i in range(len(array)):
            array[i] = array[i].detach().cpu() if isinstance(array[i], torch.Tensor) else array[i]
        array = np.stack(array)
    # Turn a single tensor into an array
    if isinstance(array, torch.Tensor): array = array.detach().cpu().numpy()
    # Handle different dtypes
    if array.dtype == np.float64: array = array.astype(np.float32)
    if array.dtype == np.bool: array = array.astype(np.float32)
    # Turn 2D and 3D into 4D
    match len(array.shape):
        case 4: image_array = grid_image(array)
        case 3: image_array = grid_image(array[np.newaxis])
        case 2: image_array = grid_image(array[np.newaxis][np.newaxis])

    Image.fromarray(image_array, "RGB").save("debug.png")

def get_env_attributes(env) -> List[Tuple[str, Any]]:
    """ Returns a list of env attributes together with wrapped env attributes. """
    attributes = []

    def extract_attributes(obj, prefix=''):
        for key, value in obj.__dict__.items():
            attributes.append((f"{prefix}{key}", value))

        if hasattr(obj, 'env'):
            extract_attributes(obj.env, f"{prefix}env.")

    extract_attributes(env)
    return attributes

def EMMA_fixation_time(
        dist: float,
        freq = 0.1,
        execution_time = 0.07,
        K = 0.006,
        k = 0.4,
        saccade_scaling = 0.002,
        t_prep = 0.135,
    ):
    """
    Mathematical model for saccade duration in seconds from EMMA (Salvucci, 2001).
    Borrowed from https://github.com/aditya02acharya/TypingAgent/blob/master/src/utilities/utils.py.

    :param float dist: Eccentricity in visual angle.
    :param float freq: Frequency of object being encoded. How often does the object appear. Value in (0,1).
    :param float execution_time: The base time it takes to execute an eye movement, independent of distance. 
    :param float K: Scaling parameter for the encoding time.
    :param float k: Scaling parameter for the influence of the saccade distance on the encoding time.
    :param float saccade_scaling: Scaling parameter for the influence of the saccade distance on the execution time.
    :param float t_prep: Movement preparation time. If this is greater than the encoding time, no movement occurs.

    :return EMMA_breakdown: tuple containing (preparation_time, execution_time, remaining_encoding_time).
    :return total_time: Total eye movement time in seconds.
    :return moved: true if encoding time > preparation time. false otherwise.
    """
    # visual encoding time
    t_enc = K * -np.log(freq) * np.exp(k * dist)

    # if encoding time < movement preparation time then no movement
    if t_enc < t_prep:
        return (t_enc, 0, 0), t_enc, False

    # movement execution time
    t_exec = execution_time + saccade_scaling * dist
    # eye movement time (preparation time + execition time)
    t_sacc = t_prep + t_exec

    # if encoding time less then movement time
    if t_enc <= t_sacc:
        return (t_prep, t_exec, 0), t_sacc, True

    # if encoding left after movement time
    e_new = (k * -np.log(freq))
    t_enc_new = (1 - (t_sacc / t_enc)) * e_new

    return (t_prep, t_exec, t_enc_new), t_sacc + t_enc_new, True

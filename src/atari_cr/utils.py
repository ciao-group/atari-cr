"""
Borrow from stable-baselines3
Due to dependencies incompability, we cherry-pick codes here
"""
import os
import random
from typing import Any, Dict, List, Tuple, Union

from matplotlib import pyplot as plt
import numpy as np
import torch
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
        return {key: get_obs_shape(subspace)
                for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]

    else:
        raise NotImplementedError(
            f"{observation_space} observation space is not supported")

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

def get_sugarl_reward_scale_atari(game) -> float:
    base_scale = 4.0
    sugarl_reward_scale = 1/200
    if game in ["alien", "assault", "asterix", "battle_zone", "seaquest", "qbert",
                "private_eye", "road_runner"]:
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

def to_uint8(array: np.ndarray):
    """ Scales and converts a float32 array image to uint8 """
    if not array.flags["WRITEABLE"]: array = array.copy()
    if array.dtype == np.float32:
        for i in range(len(array)):
            for j in range(len(array[i])):
                if array[i,j].min() == array[i,j].max():
                    # Clip arrays with the same value everywhere between 0 and 1
                    array[i,j] = np.max([array[i,j], np.zeros(array[i,j].shape)])
                    array[i,j] = np.min([array[i,j], np.ones(array[i,j].shape)])
                else:
                    # Or normalize to be between 0 and 1
                    array[i,j] = array[i,j] - array[i,j].min()
                    array[i,j] = array[i,j] / array[i,j].max()
                # Scale to be between 0 and 255
                array[i,j] *= 255
        array = array.astype(np.uint8)
    return array

def grid_image(array: np.ndarray, line_color=[249, 171, 0], line_width=1):
    """
    Turn an array of images into one 2D image separated by colored lines.

    :param Array[n_rows, n_cols, width, height, 3] array: Structured array of images
    """
    assert len(array.shape) in [4, 5], "Only works for array of shape \
        [n_rows, n_cols, x, y, n_channels] or [n_rows, n_cols, x, y]"

    # Convert greyscale to RGB
    if len(array.shape) == 4:
        array = np.broadcast_to(array[:,:,:,:,np.newaxis], [*array.shape, 3])
    if array.shape[-1] == 1:
        array = np.broadcast_to(array, [*array.shape[:-1], 3])
    n_rows, n_cols, y, x, n_channels = array.shape

    # Convert to uint8
    array = to_uint8(array)

    # Create a new RGB array to hold the grid with separating lines
    grid_size = (y * n_rows + line_width * (n_rows - 1),
        x * n_cols + line_width * (n_cols - 1), n_channels)
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

def debug_array(array: Union[np.ndarray, torch.Tensor, List[torch.Tensor]],
                out_path = "debug.png"):
    """
    Saves a 2D, 3D or 4D greyscale array as an image under 'debug.png'.
    """
    # Turn a list of arrays or tensors into one array
    if isinstance(array, List):
        for i in range(len(array)):
            if isinstance(array[i], torch.Tensor):
                array[i] = array[i].detach().cpu()
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

    Image.fromarray(image_array, "RGB").save(out_path)

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

    :param float dist: Eccentricity in visual degrees.
    :param float freq: Frequency of object being encoded. How often does the object
        appear. Value in (0,1).
    :param float execution_time: The base time it takes to execute an eye movement,
        independent of distance.
    :param float K: Scaling parameter for the encoding time.
    :param float k: Scaling parameter for the influence of the saccade distance on the
        encoding time.
    :param float saccade_scaling: Scaling parameter for the influence of the saccade
        distance on the execution time.
    :param float t_prep: Movement preparation time. If this is greater than the encoding
        time, no movement occurs.

    :return EMMA_breakdown: tuple containing (preparation_time, execution_time,
        remaining_encoding_time).
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

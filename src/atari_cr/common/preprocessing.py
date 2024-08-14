from typing import Union, Dict
import numpy as np
from gymnasium import spaces

import stable_baselines3.common.preprocessing as sb_preprocessing

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
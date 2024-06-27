from typing import Union, Dict
import numpy as np
from gymnasium import spaces

# Taken from stable_baselines3.common.preprocessing
def get_action_dim(action_space: spaces.Space) -> Union[int, Dict]:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(
            action_space.n, int
        ), f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        return int(action_space.n)
    # NEW
    elif isinstance(action_space, spaces.Dict):
        action_dims = []
        for key in action_space: 
            action_dims.append(get_action_dim(action_space[key]))
        return action_dims
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")
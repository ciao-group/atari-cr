from functools import partial
from typing import Callable, Optional, Union, Dict
from torch import nn
import torch
from torchvision.models.vision_transformer import VisionTransformer
from tqdm import tqdm
from gymnasium import spaces

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

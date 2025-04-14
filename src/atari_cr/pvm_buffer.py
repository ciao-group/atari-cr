from collections import deque
from typing import Tuple
import copy
from PIL import Image

import numpy as np


from atari_cr.utils import grid_image


class PVMBuffer:
    """
    Persistence-of-Vision memory (Section 3.3)
    """

    def __init__(self, max_len: int, obs_size: Tuple,
                 fov_loc_size: Tuple = None, mean_pvm=False) -> None:
        self.max_len = max_len
        self.obs_size = obs_size
        self.fov_loc_size = fov_loc_size
        self.buffer = None
        self.fov_loc_buffer = None
        self.init_pvm_buffer()
        self.mean_pvm = mean_pvm


    def init_pvm_buffer(self) -> None:
        self.buffer = deque([], maxlen=self.max_len)
        self.fov_loc_buffer = deque([], maxlen=self.max_len)
        for _ in range(self.max_len):
            self.buffer.append(np.zeros(self.obs_size, dtype=np.float32))
            if self.fov_loc_size is not None: # (1, 2) or (1, 2, 3)
                self.fov_loc_buffer.append(np.zeros(self.fov_loc_size, dtype=np.float32))

    def append(self, x, fov_loc=None) -> None:
        """ :param list[Tensor[4,84,84;f32]] x """
        if self.mean_pvm:
            # Rescale the obs to be between -1 and 1
            # -1 for black, 1 for white, 0 for uncertain pixels
            x = 2 * (x - 0.5)

        assert x.shape == self.obs_size, \
            "Item size has to match the size of the pvm buffer"
        self.buffer.append(x)
        if fov_loc is not None:
            self.fov_loc_buffer.append(fov_loc)

    def copy(self):
        return copy.deepcopy(self)

    def get_obs(self, mode="stack_max") -> np.ndarray:
        # [B,PVM,C,H,W] ->
        if mode == "stack_max":
            # [B, 1, C, H, W]
            return np.amax(np.stack(self.buffer, axis=1), axis=1)
        elif mode == "stack_mean":
            # [B, 1, C, H, W]
            return np.mean(np.stack(self.buffer, axis=1), axis=1)
        elif mode == "stack":
            # [B, T, C, H, W]
            return np.stack(self.buffer, axis=1)
        elif mode == "stack_channel":
            # [B, T*C, H, W]
            return np.concatenate(self.buffer, axis=1)
        else:
            raise NotImplementedError

    def get_fov_locs(self, return_mask=False, relative_transform=False) -> np.ndarray:
        transforms = []
        if relative_transform:
            for t in range(len(self.fov_loc_buffer)):
                transforms.append(np.zeros((self.fov_loc_buffer[t].shape[0], 2, 3), dtype=np.float32)) # [B, 2, 3] B=1 usually
                for b in range(len(self.fov_loc_buffer[t])):
                    extrinsic = self.fov_loc_buffer[t][b]
                    target_extrinsic = self.fov_loc_buffer[t][-1]
                    if np.linalg.det(extrinsic):
                        extrinsic_inv = np.linalg.inv(extrinsic)
                        transform = np.dot(target_extrinsic, extrinsic_inv)
                        # 4x4 transformation matrix
                        R = transform[:3, :3] # Extract the rotation
                        tr = transform[:3, 3] # Extract the translation
                        H = R + np.outer(tr, np.array([0, 0, 1], dtype=np.float32))
                        # Assuming H is the 3x3 homography matrix
                        A = H / H[2, 2]
                        affine = A[:2, :]
                        transforms[t][b] = affine
                    else:
                        H = np.identity(3, dtype=np.float32)
                        A = H / H[2, 2]
                        affine = A[:2, :]
                        transforms[t][b] = affine
            return np.stack(transforms, axis=1) # [B, T, 2, 3]
        else:
            return np.stack(self.fov_loc_buffer, axis=1)
        #[B, T, *fov_locs_size], maybe [B, T, 2] for 2d or [B, T, 4, 4] for 3d

    def to_png(self, out_file = "output/debug/pvm.png"):
        """
        Save the current content of the buffer as an image
        """
        self.to_img().save(out_file)

    def to_img(self):
        # Convert raw buffer to a grid representation
        line_width = 1
        raw_grid = grid_image(self._rgb_buffer()[0], line_width=line_width)

        # Get the pvm observation as a grid representation of the same size
        pvm_obs = self.get_obs("stack_mean" if self.mean_pvm else "stack_max")
        pvm_grid = grid_image(self._greyscale_to_rgb(pvm_obs), line_width=line_width)
        padded_pvm_grid = np.zeros(raw_grid.shape)
        padded_pvm_grid[:pvm_grid.shape[0], :, :] = pvm_grid

        # Put them underneath each other in another grid
        full_grid = grid_image(
            np.expand_dims(np.stack([raw_grid, padded_pvm_grid]), axis=1),
            line_color=[0, 255, 0],
            line_width=line_width
        )

        # Cut off the padding
        height_to_cut = (self.max_len - 1) * (pvm_obs.shape[3] + line_width)
        final_grid = full_grid[:-height_to_cut, :, :]

        return Image.fromarray(final_grid, mode="RGB")

    def _greyscale_to_rgb(self, x: np.ndarray):
        scaled_array = (x * 255).astype(np.uint8)
        return np.repeat(scaled_array[..., np.newaxis], 3, axis=-1)

    def _rgb_buffer(self):
        return self._greyscale_to_rgb(np.stack(self.buffer, axis=1))

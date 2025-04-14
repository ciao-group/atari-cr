import os
import numpy as np
import cv2
from PIL import Image

from atari_cr.agents.dqn_atari_cr.crdqn import CRDQN
from atari_cr.pvm_buffer import PVMBuffer
from atari_cr.utils import debug_array
from atari_cr.foveation import Fovea

pacman_sample = [
    "data/Atari-HEAD/ms_pacman/52_RZ_2394668_Aug-10-14-52-42/RZ_2394668_42.png",
    "data/Atari-HEAD/ms_pacman/52_RZ_2394668_Aug-10-14-52-42/RZ_2394668_43.png",
    "data/Atari-HEAD/ms_pacman/52_RZ_2394668_Aug-10-14-52-42/RZ_2394668_44.png",
    "data/Atari-HEAD/ms_pacman/52_RZ_2394668_Aug-10-14-52-42/RZ_2394668_45.png",
    "data/Atari-HEAD/ms_pacman/52_RZ_2394668_Aug-10-14-52-42/RZ_2394668_46.png",
    "data/Atari-HEAD/ms_pacman/52_RZ_2394668_Aug-10-14-52-42/RZ_2394668_47.png",
    "data/Atari-HEAD/ms_pacman/52_RZ_2394668_Aug-10-14-52-42/RZ_2394668_48.png",
    "data/Atari-HEAD/ms_pacman/52_RZ_2394668_Aug-10-14-52-42/RZ_2394668_49.png",
    "data/Atari-HEAD/ms_pacman/52_RZ_2394668_Aug-10-14-52-42/RZ_2394668_50.png",
    "data/Atari-HEAD/ms_pacman/52_RZ_2394668_Aug-10-14-52-42/RZ_2394668_51.png",
    "data/Atari-HEAD/ms_pacman/52_RZ_2394668_Aug-10-14-52-42/RZ_2394668_52.png",
    "data/Atari-HEAD/ms_pacman/52_RZ_2394668_Aug-10-14-52-42/RZ_2394668_53.png",
    "data/Atari-HEAD/ms_pacman/52_RZ_2394668_Aug-10-14-52-42/RZ_2394668_54.png",
    "data/Atari-HEAD/ms_pacman/52_RZ_2394668_Aug-10-14-52-42/RZ_2394668_55.png",
    "data/Atari-HEAD/ms_pacman/52_RZ_2394668_Aug-10-14-52-42/RZ_2394668_56.png",
    "data/Atari-HEAD/ms_pacman/52_RZ_2394668_Aug-10-14-52-42/RZ_2394668_57.png",
    "data/Atari-HEAD/ms_pacman/52_RZ_2394668_Aug-10-14-52-42/RZ_2394668_58.png",
    "data/Atari-HEAD/ms_pacman/52_RZ_2394668_Aug-10-14-52-42/RZ_2394668_59.png",
    "data/Atari-HEAD/ms_pacman/52_RZ_2394668_Aug-10-14-52-42/RZ_2394668_60.png",
    "data/Atari-HEAD/ms_pacman/52_RZ_2394668_Aug-10-14-52-42/RZ_2394668_61.png",
]

if __name__ == "__main__":
    # Get frames
    pacman_sample = np.stack( # [20,210,160,3]
        [Image.open(path) for path in pacman_sample])

    # Sort into frame stacks
    pacman_sample = np.stack([cv2.resize(img, (84,84)) for img in pacman_sample])
    pacman_sample = pacman_sample.reshape((5,4,84,84,3))
    debug_array(pacman_sample[...,0])

    # Apply foveation
    fov = Fovea("window", (26,26))
    def apply_fov(i: int, j: int, fixations: list[tuple[int,int]]):
        """
        Args:
            i (int): color channel
            j (int): pvm axis
        """
        fov_stack = fov.apply(pacman_sample[j,...,i].astype(float) / 255, fixations)
        pacman_sample[j,...,i] = (fov_stack * 255).astype(np.uint8)
    for i in range(pacman_sample.shape[-1]): # For every channel
        apply_fov(i, 0, [(36,47)])
        apply_fov(i, 1, [(15,68)])
        apply_fov(i, 2, [(5,5), (68,15)])
        apply_fov(i, 3, [(15,68)])
        apply_fov(i, 4, [(15,5)])
    out_dir = "output/graphs/pvm_showcase"
    fov_dir = f"{out_dir}/foveated"
    os.makedirs(out_dir, exist_ok=True)
    for i,stack in enumerate(pacman_sample):
        stack_dir = f"{fov_dir}/stack{i}"
        os.makedirs(stack_dir, exist_ok=True)
        for j,img in enumerate(stack):
            Image.fromarray(img).save(f"{stack_dir}/{j}.png")

    # Load into PVM buffer
    pvm_buffer = PVMBuffer(3, (3,4,84,84)) # [C,frame_stack,W,H]
    for stack in pacman_sample:
        pvm_buffer.append(stack.transpose((3,0,1,2)))

    # Get PVM output
    pvm_dir = "output/graphs/pvm_showcase/pvm"
    os.makedirs(pvm_dir, exist_ok=True)
    for i,img in enumerate(pvm_buffer.get_obs().transpose((1,2,3,0))):
        img = Image.fromarray(img)
        img.save(f"{pvm_dir}/{i}.png")

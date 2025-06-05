from PIL import Image
import os

import cv2
from atari_cr.foveation import Fovea, pyramids
import numpy as np
import polars as pl
from matplotlib import pyplot as plt

from atari_cr.graphs.common import CMAP
from atari_cr.utils import debug_array
from atari_cr.agents.dqn_atari_cr.crdqn import CRDQN

img = np.array(Image.open("data/Atari-HEAD/"
    "ms_pacman/52_RZ_2394668_Aug-10-14-52-42/RZ_2394668_42.png"))
img = img.astype(np.float64) / 255
out_dir = "output/graphs/fovs"
os.makedirs(out_dir, exist_ok=True)
multiple_out_dir = f"{out_dir}/multiple"
os.makedirs(multiple_out_dir, exist_ok=True)
fixations = np.array([[36,36], [57,15], [68,68]]) * 3
fixation = fixations[0]

# Match the visual info of the fovs
info_dir = f"{out_dir}/info"
os.makedirs(info_dir, exist_ok=True)
fovs = ["window", "window_periph", "exponential"]
sizes = np.arange(0, 102, 2)
visual_infos = np.zeros((len(fovs),len(sizes)))
sensory_action_space = CRDQN.create_sensory_action_set((84,84), 8, 8)
for i, fov in enumerate(fovs):
    for j, size in enumerate(sizes):
        visual_infos[i,j] = \
            Fovea(fov, (size,size)).get_visual_info(84, 84, sensory_action_space)
exp_info = visual_infos[2,0] # 0.1566
window_size = sizes[np.argmin(np.abs(visual_infos[0] - exp_info))] # 26
periph_size = sizes[np.argmin(np.abs(visual_infos[1] - exp_info))] # 20
visual_infos = pl.DataFrame({fovs[i]: visual_infos[i] for i in range(len(fovs))})

# Exponential fov
plt.ylim(0,1)
plt.xlabel("Fovea size")
plt.ylabel("Relative visual information")
plt.plot(sizes, visual_infos["exponential"], alpha=0.75, lw=2)
plt.savefig(f"{info_dir}/exponential.png", bbox_inches='tight', pad_inches=0.1,)
# Windowed fovs
for fov, size in zip(["window", "window_periph"], [window_size, periph_size]):
    plt.clf()
    plt.ylim(0.,1.)
    plt.xlabel("Fovea size")
    plt.ylabel("Relative visual information")
    plt.plot(sizes, visual_infos[fov], alpha=0.75, lw=2)
    plt.plot(sizes, visual_infos["exponential"], alpha=0.75, color=CMAP[1], lw=2)
    # Vertical line for matching x value
    plt.axvline(x=size, color=CMAP[1], linestyle='--', label=f'x = {size}', alpha=0.5, lw=2)
    plt.savefig(f"{info_dir}/{fov}.png", bbox_inches='tight', pad_inches=0.1)
print(visual_infos)

# Original images with marks
marked_img = (img.copy() * 255).astype(np.uint8)
for i, fix in enumerate(fixations):
    marked_img = cv2.drawMarker(marked_img, fix, [0,255,0], 1, 15, 2)
    if i == 0:
        Image.fromarray(marked_img, "RGB").save(f"{out_dir}/original.png")
Image.fromarray(marked_img, "RGB").save(f"{multiple_out_dir}/original.png")

fovs = [("window", window_size*3), ("window_periph", periph_size*3), ("exponential", 0)]
for type, size in fovs:
    fov = Fovea(type, [size,size])
    fov_img = fov.apply(img.copy().transpose(2,0,1), [fixation])
    Image.fromarray((fov_img.transpose(1,2,0) * 255).astype(np.uint8), "RGB").save(
        f"{out_dir}/{type}.png")
    multiple_fov_img = fov.apply(img.copy().transpose(2,0,1), fixations)
    Image.fromarray((multiple_fov_img.transpose(1,2,0) * 255).astype(np.uint8), "RGB").save(
        f"{multiple_out_dir}/{type}.png")

# Pyramids
pyramid_dir = f"{out_dir}/pyramids"
os.makedirs(pyramid_dir, exist_ok=True)
for i, pyramid in enumerate(pyramids(img)):
    Image.fromarray((pyramid * 255).astype(np.uint8), "RGB").save(
        f"{pyramid_dir}/{i}.png")

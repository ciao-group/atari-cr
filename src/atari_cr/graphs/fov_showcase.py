from PIL import Image
import os

import cv2
from atari_cr.foveation import Fovea, pyramids
import numpy as np

from atari_cr.utils import debug_array
from atari_cr.agents.dqn_atari_cr.crdqn import CRDQN

def get_visual_info(fov: Fovea, w: int, h: int, sensory_action_space: list[list[int]]):
    visual_infos = np.empty(len(sensory_action_space))
    for i, fov_loc in enumerate(sensory_action_space):
        _, visual_infos[i] = fov.apply(np.zeros((4,w,h)), [fov_loc])
    return visual_infos.mean()

img = np.array(
    Image.open("/home/niko/Repos/atari-cr/tests/assets/ms_pacman.png"))
img = img.astype(np.float64) / 255
out_dir = "output/graphs/fovs"
os.makedirs(out_dir, exist_ok=True)
fixation = [32*3,32*3]

# Match the visual info of the fovs
fovs = ["window", "window_periph", "exponential"]
sizes = np.arange(0, 102, 2)
visual_infos = np.zeros((len(fovs),len(sizes)))
sensory_action_space = CRDQN.create_sensory_action_set((84,84), 8, 8)
for i, fov in enumerate(fovs):
    for j, size in enumerate(sizes):
        visual_infos[i,j] = get_visual_info(
            Fovea(fov, (size,size)), 84, 84, sensory_action_space)
exp_info = visual_infos[2,0] # 0.1566
window_size = sizes[np.argmin(np.abs(visual_infos[0] - exp_info))] # 34
periph_size = sizes[np.argmin(np.abs(visual_infos[1] - exp_info))] # 26
print(f"Exponential Info: {exp_info:}, Window Size: {window_size}, "
      f"Peripheral Window Size: {periph_size}")

# Original image with marks
marked_img = cv2.drawMarker((img*255).astype(np.uint8), fixation, [0,255,0], 1, 15, 2)
Image.fromarray(marked_img, "RGB").save(f"{out_dir}/original.png")

fovs = [("window", window_size*3), ("window_periph", periph_size*3), ("exponential", 0)]
infos = []
for type, size in fovs:
    fov = Fovea(type, [size,size])
    fov_img, visual_info = fov.apply(img.copy().transpose(2,0,1), [fixation])
    infos.append(visual_info)

    Image.fromarray((fov_img.transpose(1,2,0) * 255).astype(np.uint8), "RGB").save(
        f"{out_dir}/{type}.png")
print(infos)

# Pyramids
pyramid_dir = f"{out_dir}/pyramids"
os.makedirs(pyramid_dir, exist_ok=True)
for i, pyramid in enumerate(pyramids(img)):
    Image.fromarray((pyramid * 255).astype(np.uint8), "RGB").save(
        f"{pyramid_dir}/{i}.png")

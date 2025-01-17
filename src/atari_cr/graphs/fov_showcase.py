from PIL import Image
import os

import cv2
from atari_cr.foveation import Fovea, pyramids
import numpy as np
from atari_cr.utils import debug_array

img = np.array(
    Image.open("/home/niko/Repos/atari-cr/tests/assets/ms_pacman.png"))
img = img.astype(np.float64) / 255
out_dir = "output/graphs/fovs"
os.makedirs(out_dir, exist_ok=True)
fixation = [32*3,32*3]

# Original image with marks
marked_img = cv2.drawMarker((img*255).astype(np.uint8), fixation, [0,255,0], 1, 15, 2)
Image.fromarray(marked_img, "RGB").save(f"{out_dir}/original.png")

fovs = [("window", 37*3), ("window_periph", 31*3), ("exponential", 0)]
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

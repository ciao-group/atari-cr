from PIL import Image
import os

import cv2
from atari_cr.foveation import Fovea
import numpy as np
from atari_cr.utils import debug_array

img = np.array(
    Image.open("/home/niko/Repos/atari-cr/tests/assets/ms_pacman.png").convert("L"))
img = cv2.resize(img, [84,84])
img = img.astype(np.float64) / 255
out_dir = "output/graphs/fovs"
os.makedirs(out_dir, exist_ok=True)
fixation = [32,32]

# Original image with marks
marked_img = cv2.drawMarker((img * 255).astype(np.uint8), fixation, [255,0,0], 1, 5)
Image.fromarray(marked_img, "L").save(f"{out_dir}/original.png")

fovs = [("window", 37), ("window_periph", 31), ("exponential", 0)]
infos = []
for type, size in fovs:
    fov = Fovea(type, [size,size])
    fov_img, visual_info = fov.apply(img[np.newaxis,...], [fixation])
    infos.append(visual_info)

    Image.fromarray((fov_img[0] * 255).astype(np.uint8) , "L").save(
        f"{out_dir}/{type}.png")
print(infos)

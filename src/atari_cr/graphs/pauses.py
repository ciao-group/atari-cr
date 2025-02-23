import os
import shutil
import numpy as np
from PIL import Image

from atari_cr.graphs.common import Run
from atari_cr.models import EpisodeRecord

run = Run("output/good_ray_runs/exp_2_3m_video_2025-02-20_11-29-23")
out_dir = "output/graphs/pauses"
shutil.rmtree(out_dir)
os.makedirs(out_dir)

# Get the first trial matching a given env
envs = ["asterix", "seaquest", "hero"]
for env in envs:
    trial = [e.path for e in os.scandir(run.path) if env in e.path][0]

    # First eval run in the trial
    record = EpisodeRecord.load(
        [e.path for e in os.scandir(f"{trial}/recordings")][0])

    # Create output dir for every env
    env_dir = f"{out_dir}/{env}"
    os.makedirs(env_dir, exist_ok=True)

    pause_inds = np.where(record.annotations["pauses"])[0]
    # Filter for pauses of length two
    pause_inds = [i for i in pause_inds if (i + 1) in pause_inds]
    for i in pause_inds:
        # Save the six previous images
        frame_dir = f"{env_dir}/{i}"
        os.makedirs(frame_dir, exist_ok=True)
        for j in range(i-6, i+7, 3):
            Image.fromarray(record.frames[j], mode="RGB").save(f"{frame_dir}/{j}.png")

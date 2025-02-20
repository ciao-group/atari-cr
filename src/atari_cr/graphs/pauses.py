import os
import numpy as np
from PIL import Image

from atari_cr.graphs.common import Run
from atari_cr.models import EpisodeRecord

run = Run("output/good_ray_runs/exp_2_3m_video_2025-02-18_19-39-37")
out_dir = "output/graphs/pauses"
os.makedirs(out_dir, exist_ok=True)

# Get the first trial matching a given env
envs = ["asterix", "seaquest", "hero"]
for env in envs:
    trial = [e.path for e in os.scandir(run.path) if env in e.path][0]

    # First eval run in the trial
    record = EpisodeRecord.load(
        [e.path for e in os.scandir(f"{trial}/recordings")][0])

    # Create output dir for every env
    env_dir = f"{out_dir}/{env}"
    os.makedirs(env_dir)

    paused_frames = record.frames[np.where(record.annotations["pauses"])[0]]
    for i, frame in enumerate(paused_frames):
        Image.fromarray(frame, mode="RGB").save(f"{env_dir}/{i}.png")

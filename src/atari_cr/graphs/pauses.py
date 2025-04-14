import os
import shutil
import cv2
import numpy as np
from PIL import Image
import polars as pl

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

    # Coord Scaling to 256x256
    record.annotations = record.annotations.with_columns([
        pl.col("fov_x") * 3,
        pl.col("fov_y") * 3,
        pl.col("sensory_action_x") * 3,
        pl.col("sensory_action_y") * 3,
    ])

    pause_inds = np.where(record.annotations["pauses"])[0]
    # Filter for pauses of length two
    pause_inds = np.array([i for i in pause_inds if (i + 1) in pause_inds])
    for i in pause_inds:
        # Save the six previous images
        frame_dir = f"{env_dir}/{i}"
        os.makedirs(frame_dir, exist_ok=True)
        for j in range(i-6, i+7, 3):
            # Draw red cross on gaze loc; green crosses on three desired locs
            frame = record.frames[j]
            fov_loc = record.annotations[j, ["fov_x", "fov_y"]].row(0)

            # Collect all sensory actions for the frame and following frames, if it was
            # a pause
            sensory_actions = []
            k = j
            while record.annotations[k, "pauses"]:
                sensory_actions.append(
                    record.annotations[k, ["sensory_action_x", "sensory_action_y"]]
                        .row(0)
                )
                k += 1
            sensory_actions.append(
                record.annotations[k, ["sensory_action_x", "sensory_action_y"]].row(0)
            )

            for s in sensory_actions:
                cv2.drawMarker(frame, s, (255,0,0), 1, 15, 2) # Red sensory_action
            cv2.drawMarker(frame, fov_loc, (0,255,0), 1, 15, 2) # Green fov_loc
            Image.fromarray(frame, mode="RGB").save(f"{frame_dir}/{j}.png")

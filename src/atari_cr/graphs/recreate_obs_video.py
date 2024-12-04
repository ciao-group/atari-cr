""" Saves the obs video without the drawn fovea """
import os

import numpy as np

from atari_cr.atari_head.utils import open_mp4_as_frame_list
from atari_cr.models import EpisodeRecord


records_path = "/tmp/ray/session_2024-12-03_13-00-04_962907_498740/artifacts/2024-12-03_13-00-06/lambda_2024-12-03_13-00-04/working_dirs/lambda_22f43_00005_5_seed=1,sensory_action_space_quantization=16_2024-12-03_13-00-06/output/runs/tuning/ms_pacman/recordings"
for record_name in os.listdir(records_path):
    record_path = os.path.join(records_path, record_name)
    video_path = os.path.join(record_path, "video.mp4")
    obs_path = os.path.join(record_path, "obs.mp4")

    observations = open_mp4_as_frame_list(obs_path) # -> [256,512,3]
    frames = open_mp4_as_frame_list(video_path)
    for i in range(len(observations)):
        observations[i][:,256:,:] = frames[i]

    EpisodeRecord._save_video(np.stack(observations), obs_path)

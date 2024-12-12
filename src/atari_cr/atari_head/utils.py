import os
import cv2
import numpy as np
import polars as pl
import torch

# Screen Size in visual degrees: 44,6 x 28,5
# Visual Degrees per Pixel with 84 x 84 pixels: 0,5310 x 0,3393
VISUAL_DEGREE_SCREEN_SIZE = (44.6, 28.5)

def transform_to_proper_csv(game_dir: str):
    """
    Transforms the pseudo csv format used by Atari-HEAD to proper csv

    :param str game_dir: The directory containing files for one game.
        Obtained by unzipping \\<game\\>.zip
    """
    csv_files = list(filter(
        lambda file_name: ".txt" in file_name, os.listdir(game_dir)))
    for file_name in csv_files:

        # Read the original file
        file_path = f"{game_dir}/{file_name}"
        with open(file_path, "r") as f:
            lines = f.readlines()

        data = []
        for line in lines[1:]:
            # Put the gaze positions into a list of tuples instead of a flat list
            tupled_gaze_positions = []
            gaze_positions = line.split(",")[6:]
            for i in range(len(gaze_positions) // 2):
                x_coord = gaze_positions[2 * i]
                y_coord = gaze_positions[2 * i + 1]
                tupled_gaze_positions.append((x_coord, y_coord))

            # Append a new row to the data
            data.append([
                *line.split(",")[:6],
                tupled_gaze_positions
            ])

        # Export the data to csv and delete the original files
        df = pl.DataFrame(data, columns=[
            "frame_id",
            "episode_id",
            "score",
            "duration",
            "unclipped_reward",
            "action",
            "gaze_positions"
        ])
        df.write_csv(".".join(file_path.split(".")[:-1]) + ".csv", index=False)
        os.remove(file_path)

def open_mp4_as_frame_list(path: str):
    video = cv2.VideoCapture(path)

    frames = []
    while True:
        success, frame = video.read()

        if success:
            frames.append(frame)
        else:
            break

    video.release()
    return frames

def preprocess(frame: np.ndarray):
    """
    Image preprocessing function from IL-CGL.
    Warp frames to 84x84 as done in the Nature paper and later work.

    :param np.ndarray frame: uint8 greyscale frame loaded using `cv2.imread`
    """
    width = 84
    height = 84
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return torch.Tensor(frame / 255.0)

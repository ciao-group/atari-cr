import os
import cv2
import numpy as np
import polars as pl
import torch
from PIL import Image

from atari_cr.common.models import RecordBuffer
from atari_cr.common.utils import grid_image

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
            "duration(ms)",
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

def debug_recording(recordings_path: str):
    """
    :param str recordings_path: Path to the agent's eval data,
        containing images and associated gaze positions
    """
    # Get the recording data of the first recording as a dict
    file = list(filter(lambda x: x.endswith(".pt"), os.listdir(recordings_path)))[0]
    data: RecordBuffer = torch.load(
        os.path.join(recordings_path, file), weights_only=False)

    # Extract a list of frames and a list of gazes
    frames = open_mp4_as_frame_list(data["rgb"])
    actions = data["action"]
    assert len(frames) == len(actions)

    for frame, action in zip(frames, actions):

        boxing_pause_action = 18
        if action == boxing_pause_action:
            # Write "pause" on the frame
            text = "pause"
            position = (10, 20)
            font = cv2.FONT_HERSHEY_COMPLEX
            font_scale = 0.3
            color = (255, 0, 0)
            thickness = 1
            frame = cv2.putText(
                frame, text, position, font, font_scale, color, thickness)

    # Display images in a grid
    grid = np.array(frames[:16])
    grid = grid.reshape([4, 4, *grid.shape[1:]])
    grid = grid_image(grid)
    Image.fromarray(grid).save("debug.png")

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

import os
import cv2
import numpy as np
import polars as pl
import torch
import json

SCREEN_SIZE = (84, 84)
# Atari-HEAD Screen Size in visual degrees: 44,6 x 28,5
# Atari-HEAD Screen Size in pixels: 1280 x 840
# Visual Degrees per Pixel with 84 x 84 pixels: 0,5310 x 0,3393
VISUAL_DEGREE_SCREEN_SIZE = (44.6, 28.5) # (W,H)
VISUAL_DEGREES_PER_PIXEL = np.array(VISUAL_DEGREE_SCREEN_SIZE) / np.array(SCREEN_SIZE)

def transform_to_proper_csv(game_dir: str):
    """
    Transforms the pseudo csv format used by Atari-HEAD to proper csv

    :param str game_dir: The directory containing files for one game.
        Obtained by unzipping \\<game\\>.zip
    """
    print(f"Tranforming .txt files in {game_dir}")
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
                x_coord = float(gaze_positions[2 * i])
                y_coord = float(gaze_positions[2 * i + 1])
                tupled_gaze_positions.append((x_coord, y_coord))

            # Append a new row to the data
            data.append([
                *[None if x == "null" else x for x in line.split(",")[:6]],
                json.dumps(tupled_gaze_positions),
            ])

        # Export the data to csv and delete the original files
        # The last row only contains null values
        df = pl.DataFrame(data[:-1], orient="row", schema={
            "frame_id": pl.String,
            "episode_id": pl.Int32,
            "score": pl.Float32,
            "duration(ms)": pl.Float32,
            "unclipped_reward": pl.Float32,
            "action": pl.Int32,
            "gaze_positions": pl.String,
        })
        df.write_csv(".".join(file_path.split(".")[:-1]) + ".csv")
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

def save_video(frames: np.ndarray, video_path: str, greyscale = False):
    """ Saves a numpy array as .mp4 """
    size = frames[0].shape[:2][::-1]
    fps = 12 # 60 / frame_skip
    video_writer = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size,
        isColor=not greyscale)
    if frames.dtype == np.float32:
        frames = (frames * 255).astype(np.uint8)
    for frame in frames:
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video_writer.release()

def preprocess(frame: np.ndarray):
    """
    Image preprocessing function from IL-CGL.
    Warp frames to 84x84 as done in the Nature paper and later work.

    :param Array[W,H,3; u8] frame: uint8 greyscale frame loaded using `cv2.imread`
    :returns Array[84,84; f32]:
    """
    width = 84
    height = 84
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return torch.Tensor(frame / 255.0)

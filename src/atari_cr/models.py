from enum import Enum
import os
from typing import TypedDict

import cv2
import numpy as np
import polars as pl
import yaml

class SensoryActionMode(Enum):
    ABSOLUTE = 1
    RELATIVE = 2

    @staticmethod
    def from_string(s: str):
        match(s.lower()):
            case "absolute":
                return SensoryActionMode.ABSOLUTE
            case "relative":
                return SensoryActionMode.RELATIVE
            case _:
                raise ValueError("Invalid sensory action mode")

    def __str__(self):
        match(self):
            case SensoryActionMode.ABSOLUTE:
                return "absolute"
            case SensoryActionMode.RELATIVE:
                return "relative"

class EpisodeInfo(TypedDict):
    """
    :var int reward_wo_sugarl: Raw raw reward from the base env
    """
    reward: int
    raw_reward: int
    reward_wo_sugarl: int
    pauses: int
    prevented_pauses: int
    no_action_pauses: int
    saccade_cost: float

    @staticmethod
    def new():
        """ Creates a new dict with default values. """
        info: EpisodeInfo = { k: v() for k, v in EpisodeInfo.__annotations__.items() }
        return info

class StepInfo(EpisodeInfo):
    episode_info: EpisodeInfo
    timestep: int
    fov_loc: tuple[int, int]
    motor_action: int
    sensory_action: tuple[int, int]
    done: bool
    truncated: bool
    consecutive_pauses: bool

    @staticmethod
    def new():
        """ Creates a new dict with default values. """
        info: StepInfo = { k: v() for k, v in StepInfo.__annotations__.items() }
        info["episode_info"] == EpisodeInfo.new()
        return info

class EpisodeArgs(TypedDict):
    fov_size: tuple[int, int]

class EpisodeRecord():
    """
    Recording of one game episode containing annotated frames

    :param Array[N,256,256,3] frames:
    :param pl.DataFrame annotations: DataFrame containing StepInfo objects as rows
    :param EpisodeArgs args:
    """
    def __init__(self, frames: np.ndarray, annotations: pl.DataFrame,
                 args: EpisodeArgs):
        assert len(frames) == len(annotations), "len of annotations needs to match len \
            of frames as every frame should have an annotation"
        self.frames = frames
        self.annotations = annotations
        self.args = args

    @staticmethod
    def _file_paths(save_dir: str):
        """ Returns paths to save the files making up the EpisodeRecord """
        os.makedirs(save_dir, exist_ok=True)
        video_path = f"{save_dir}/video.mp4"
        annotation_path = f"{save_dir}/annotations.csv"
        args_path = f"{save_dir}/args.yaml"
        return video_path, annotation_path, args_path

    def save(self, save_dir: str, draw_focus = False, draw_pauses = False):
        """
        Saves the record_buffer to an mp4 file and a metadata file.

        :param bool draw_focus: Whether to draw the fovea onto the frames in the video
        :param bool draw_focus: Whether to draw a cumulative pause count onto the frames
            in the video
        """
        # Output paths
        video_path, annotation_path, args_path = EpisodeRecord._file_paths(save_dir)

        # Writer that iteratively saves frames in an mp4 file
        size = self.frames[0].shape[:2][::-1]
        fps = 30
        video_writer = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

        # Style settings
        color = (255, 0, 0)
        thickness = 1

        if draw_focus:
            # The fov_loc is set in a 84x84 grid; the video output is 256x256
            # This scales it down
            COORD_SCALING = 256 / 84
            fov_locs = np.stack(self.annotations["fov_x"].to_numpy(),
                                self.annotations["fov_y"].to_numpy())
            fov_locs *= COORD_SCALING
            fov_size = np.array(self.args["fov_size"]) * COORD_SCALING

        # Save the frames as mp4
        for i, frame in enumerate(self.frames):

            if draw_focus:
                top_left = fov_locs[i]
                bottom_right = top_left + fov_size
                frame = cv2.rectangle(
                    frame, top_left, bottom_right, color, thickness)

            if draw_pauses:
                text = f"Pauses: {self.annotations[i, 'episode_info']['pauses']}"
                position = (10, 20)
                font = cv2.FONT_HERSHEY_COMPLEX
                font_scale = 0.3
                frame = cv2.putText(
                    frame, text, position, font, font_scale, color, thickness)

            video_writer.write(frame)
        video_writer.release()

        # Safe annotations and args
        self.annotations.write_csv(annotation_path)
        with open(args_path, "w") as f:
            yaml.safe_dump(self.args, f)

    @staticmethod
    def load(save_dir: str):
        # Input paths
        video_path, annotation_path, args_path = EpisodeRecord._file_paths(save_dir)

        # Load mp4 file as list of numpy frames
        frames = []
        vid_capture = cv2.VideoCapture(video_path)
        ret, frame = vid_capture.read()
        while(ret):
            frames.append(frame)
            ret, frame = vid_capture.read()
        frames = np.stack(frames)

        # Load the annoations as a polars DataFrame
        annotations = pl.read_csv(annotation_path)

        # Load the args from yaml
        with open(args_path, "r") as f:
            args = yaml.safe_load(f)

        return EpisodeRecord(frames, annotations, args)

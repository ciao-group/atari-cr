import os
from typing import Literal, NamedTuple, Optional, TypeAlias, TypedDict

import cv2
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import yaml

from atari_cr.atari_head.durations import BINS, get_histogram
from atari_cr.atari_head.utils import open_mp4_as_frame_list

class EpisodeInfo(TypedDict):
    """
    :var int reward_wo_sugarl: Raw raw reward from the base env
    """
    reward: int
    raw_reward: int # Reward without pause penalties
    reward_wo_sugarl: int # Reward without sugarl term
    pauses: int
    prevented_pauses: int
    no_action_pauses: int
    saccade_cost: float
    truncated: int
    emma_time: float

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
    consecutive_pauses: bool

    @staticmethod
    def new():
        """ Creates a new dict with default values. """
        info: StepInfo = { k: v() for k, v in StepInfo.__annotations__.items() }
        info["episode_info"] == EpisodeInfo.new()
        return info

class EpisodeArgs(TypedDict):
    fov_size: tuple[int, int]

class RecordBuffer(TypedDict):
    """ Implementation of an episode record in the original sugarl code """
    rgb: list
    state: list
    action: list
    reward: list
    done: list
    truncated: list
    info: list
    return_reward: list
    fov_size: list
    fov_loc: list

class EpisodeRecord():
    """
    Recording of one game episode containing annotated frames

    :param Array[N,256,256,3] frames:
    :param pl.DataFrame annotations: DataFrame containing StepInfo objects as rows
    :param EpisodeArgs args:
    :param Array[N,84,84] obs: The last frames of the agent's observed frame stack on
        each step
    """
    def __init__(self, frames: np.ndarray, annotations: pl.DataFrame,
                 args: EpisodeArgs, obs: Optional[np.ndarray] = None):
        assert len(frames) == len(annotations), "len of annotations needs to match len \
            of frames as every frame should have an annotation"
        self.frames = frames
        self.annotations = annotations
        self.args = args
        self.obs = obs

    @staticmethod
    def annotations_from_step_infos(step_infos: list[StepInfo]):
        """
        Casts a list of StepInfos into a polars dataframe for representations
        in the EpisodeRecord
        """
        return pl.DataFrame(step_infos).with_columns(
            pl.col("episode_info").struct.rename_fields(
                [f"cum_{col}" for col in EpisodeInfo.__annotations__.keys()])
            ).unnest("episode_info").with_columns([
                # Split fov_loc and sensory_action into smaller columns
                # To make the df exportable via csv later
                pl.col("fov_loc").map_elements(
                    lambda a: a[0], pl.Int32).alias("fov_x"),
                pl.col("fov_loc").map_elements(
                    lambda a: a[1], pl.Int32).alias("fov_y"),
                pl.col("sensory_action").map_elements(
                    lambda a: a[0], pl.Int32).alias("sensory_action_x"),
                pl.col("sensory_action").map_elements(
                    lambda a: a[1], pl.Int32).alias("sensory_action_y"),
            ]).drop(["fov_loc", "sensory_action"])

    @staticmethod
    def _file_paths(save_dir: str):
        """ Returns paths to save the files making up the EpisodeRecord """
        os.makedirs(save_dir, exist_ok=True)
        video_path = f"{save_dir}/video.mp4"
        annotation_path = f"{save_dir}/annotations.csv"
        args_path = f"{save_dir}/args.yaml"
        obs_path = f"{save_dir}/obs.mp4"
        return video_path, annotation_path, args_path, obs_path

    @staticmethod
    def _save_video(frames: np.ndarray, video_path: str, greyscale = False):
        """ Saves a numpy array as .mp4 """
        size = frames[0].shape[:2][::-1]
        fps = 30
        video_writer = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size,
            isColor=not greyscale)
        if greyscale and frames.dtype == np.float32:
            frames = (frames * 255).astype(np.uint8)
        for frame in frames:
            video_writer.write(frame)
        video_writer.release()

    @staticmethod
    def from_record_buffer(buffer: RecordBuffer):
        # Read frames directly or from an mp4 file
        if isinstance(buffer["rgb"], str):
            frames = np.stack(open_mp4_as_frame_list(buffer["rgb"]))
        else: frames = np.stack(buffer["rgb"])

        # Map keys from the buffer to StepInfo
        step_infos = []
        for i in range(len(frames)):
            step_info = StepInfo.new()
            step_info["reward"] = buffer["reward"][i]
            step_info["raw_reward"] = buffer["reward"][i]
            step_info["done"] = buffer["done"][i]
            step_info["truncated"] = buffer["truncated"][i]
            step_info["fov_loc"] = tuple(buffer["info"][i]["fov_loc"])
            step_info["episode_info"] = EpisodeInfo.new()
            step_infos.append(step_info)
        annotations = EpisodeRecord.annotations_from_step_infos(step_infos)
        annotations = annotations.with_columns([
            # The sensory_action is set to the next fov location
            pl.col("fov_x").shift(-1).alias("sensory_action_x"),
            pl.col("fov_y").shift(-1).alias("sensory_action_y"),
            # Manually calculate the EpisodeInfo columns
            pl.col("reward").cum_sum().alias("cum_reward"),
            pl.col("raw_reward").cum_sum().alias("cum_raw_reward"),
        ])

        args = { "fov_size": buffer["fov_size"][0] }
        return EpisodeRecord(frames, annotations, args)

    def save(self, save_dir: str, draw_focus = False, with_obs = False):
        """
        Saves the record_buffer to an mp4 file and a metadata file.

        :param bool draw_focus: Whether to draw the fovea onto the frames in the video
        """
        # Output paths
        video_path, annotation_path, args_path, obs_path = \
            EpisodeRecord._file_paths(save_dir)

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

        # Draw onto the frames and save them as mp4
        frames = []
        for i, frame in enumerate(self.frames):

            if draw_focus:
                top_left = fov_locs[i]
                bottom_right = top_left + fov_size
                frame = cv2.rectangle(
                    frame, top_left, bottom_right, color, thickness)

            # Draw a red dot on the image if the agent paused
            if self.annotations[i, 'pauses']:
                frame[247:251, 4:8, :] = np.broadcast_to(
                    np.array([0, 0, 255]), [4, 4, 3])

            frames.append(frame)
        self._save_video(np.stack(frames), video_path)

        # Safe annotations and args
        self.annotations.write_csv(annotation_path)
        with open(args_path, "w") as f:
            yaml.safe_dump(self.args, f)

        # Safe observations together with original frames
        if with_obs and self.obs is not None:
            upscaled_obs = np.stack([cv2.resize(
                    np.broadcast_to(obs[...,np.newaxis], (*obs.shape, 3)),
                    (256, 256)
                ) for obs in (self.obs * 255).astype(np.uint8)])
            self._save_video(np.concatenate([upscaled_obs, frames], axis=2), obs_path)

    @staticmethod
    def load(save_dir: str):
        # Input paths
        video_path, annotation_path, args_path, obs_path = \
            EpisodeRecord._file_paths(save_dir)

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

        # Load observations
        obs = None
        if os.path.exists(obs_path):
            with open(obs_path, "rb") as f:
                obs: np.ndarray = np.load(f)

        return EpisodeRecord(frames, annotations, args, obs)

class EvalResult(TypedDict):
    min: float
    max: float
    sum: float
    kl_div: float
    auc: float
    entropy: float

FovType: TypeAlias = Literal["window", "gaussian", "exponential"]

class TdUpdateInfo(NamedTuple):
    old_value: float
    td_target: float
    reward: float
    sugarl_penalty: float
    next_state_value: float
    loss: float

class DurationInfo:
    error: float
    mean: float
    median: float

    def __init__(self, error: float, mean: float, median: float):
        self.error = error
        self.mean = mean
        self.median = median

    @staticmethod
    def from_episodes(records: list[EpisodeRecord], env_name: str):
        durations = [0.]
        annotations = pl.concat([record.annotations for record in records])
        # Add to the gaze duration
        for pauses, emma_time in annotations["pauses", "emma_time"].iter_rows():
            durations[-1] += emma_time
            if pauses == 0:
                # Make a new duration for the next frame
                durations.append(0.)
        # Drop the last empty duration
        durations = torch.Tensor(durations[:-1])

        # Calculate histogram and distance to the expected histogram for the given game
        histogram = torch.histogram(durations, BINS).hist
        histogram = histogram / histogram.sum()
        error = F.mse_loss(histogram, get_histogram(env_name)).item()

        return DurationInfo(
            error,
            durations.mean().item(),
            durations.median().item(),
        )

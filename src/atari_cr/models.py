import os
from typing import NamedTuple, Optional, TypedDict

import cv2
import numpy as np
import polars as pl
import torch
import yaml
from scipy.stats import wasserstein_distance

from atari_cr.atari_head.durations import BINS, get_durations
from atari_cr.atari_head.utils import open_mp4_as_frame_list
from atari_cr.foveation import Fovea


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
    duration: float

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
        assert len(frames) == len(annotations) + 1, ("len of annotations needs to match"
            "len of frames as every frame should have an annotation")
        self.frames = frames
        self.annotations = annotations
        self.args = args
        self._obs = obs

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
    def _load_video(path: str):
        frames = []
        vid_capture = cv2.VideoCapture(path)
        ret, frame = vid_capture.read()
        while(ret):
            frames.append(frame)
            ret, frame = vid_capture.read()
        return np.stack(frames)

    @staticmethod
    def from_record_buffer(buffer: RecordBuffer):
        # Read frames directly or from an mp4 file
        if isinstance(buffer["rgb"], str):
            frames = np.stack(open_mp4_as_frame_list(buffer["rgb"]))
        else: frames = np.stack(buffer["rgb"])

        # Map keys from the buffer to StepInfo
        step_infos = []
        # The first entry in the record buffer is a transition into the initial state
        # without action or reward. It is excluded
        for i in range(1, len(frames)):
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

        # The fov_loc is set in a 84x84 grid; the video output is 256x256
        # This scales it down
        COORD_SCALING = 255 / 83
        fov_locs = np.concat([
                self.annotations["fov_x", "fov_y"].to_numpy(),
                self.annotations["sensory_action_x", "sensory_action_y"].to_numpy()[-1:],
            ], axis=0
        )
        fov_locs = np.rint(fov_locs.astype(np.float32) * COORD_SCALING)
        fov_size = np.rint(np.array(self.args["fov_size"]) * COORD_SCALING)

        # Draw onto the frames and save them as mp4
        frames = []
        for i, frame in enumerate(self.frames[:-1]):

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
        frames.append(self.frames[-1])
        EpisodeRecord._save_video(np.stack(frames), video_path)

        # Safe annotations and args
        self.annotations.write_csv(annotation_path)
        with open(args_path, "w") as f:
            yaml.safe_dump(self.args, f)

        # Safe observations together with original frames
        if with_obs and self._obs is not None:
            upscaled_obs = np.stack([cv2.resize(
                    np.broadcast_to(obs[...,np.newaxis], (*obs.shape, 3)),
                    (256, 256)
                ) for obs in (self._obs * 255).astype(np.uint8)])

            Fovea(self.args["fov"], fov_size).draw(frames, fov_locs)
            EpisodeRecord._save_video(np.concatenate([upscaled_obs, frames], axis=2),
                                      obs_path)

    @staticmethod
    def load(save_dir: str):
        # Input paths
        video_path, annotation_path, args_path, obs_path = \
            EpisodeRecord._file_paths(save_dir)

        # Load mp4 file as list of numpy frames
        frames = EpisodeRecord._load_video(video_path)

        # Load the annoations as a polars DataFrame
        annotations = pl.read_csv(annotation_path)

        # Load the args from yaml
        with open(args_path, "r") as f:
            args = yaml.safe_load(f)

        # Load observations
        obs = None
        if os.path.exists(obs_path):
            obs = EpisodeRecord._load_video(obs_path)
            # Cut off the part with the fully colored and visible frames
            obs = obs[:,:,:256,:]
            # Scale back from [N,256,256,3] to [N,84,84]
            new_obs = np.empty([len(obs), 84, 84])
            for i in range(len(obs)):
                new_obs[i] = cv2.cvtColor(cv2.resize(obs[i], [84,84]),
                                          cv2.COLOR_BGR2GRAY)
            obs = new_obs

        return EpisodeRecord(frames, annotations, args, obs)

    @property
    def obs(self):
        return self._obs

    @obs.setter
    def obs(self, obs: np.ndarray):
        """ :param Array[N,84,84] obs: New observations value """
        self._obs = obs
        assert len(self.obs) == len(self.frames), \
            "Number of observations must match the number of frames"

class EvalResult(TypedDict):
    min: float
    max: float
    sum: float
    kl_div: float
    auc: float
    entropy: float

class TdUpdateInfo(NamedTuple):
    old_value: float
    td_target: float
    reward: float
    sugarl_penalty: float
    next_state_value: float
    loss: float
    motor_target_max: float
    sensory_target_max: float

class DurationInfo:
    """ Durations in ms """
    error: float
    mean: float
    median: float
    hist: list[float]
    durations: list[float]

    def __init__(self, error: float, mean: float, median: float, hist: list[float],
                 durations: list[float]):
        self.error = error
        self.mean = mean
        self.median = median
        self.hist = hist
        self.durations = durations

    @staticmethod
    def from_episodes(records: list[EpisodeRecord], env_name: str):
        """ Returns None if the episode consists of only pauses """
        durations = [0.]
        annotations = pl.concat([record.annotations for record in records])
        # Add to the gaze duration
        for pauses, duration in annotations["pauses", "duration"].iter_rows():
            durations[-1] += duration
            if pauses == 0:
                # Make a new duration for the next frame
                durations.append(0.)
        # Drop the last empty duration
        durations = torch.Tensor(durations[:-1])

        # Return None if there are no durations
        if len(durations) == 0: return None

        # Calculate histogram and distance to the expected histogram for the given game
        histogram = torch.histogram(durations, BINS).hist
        histogram = histogram / histogram.sum()
        error = wasserstein_distance(durations, get_durations(env_name))

        return DurationInfo(
            error,
            durations.mean().item(),
            durations.median().item(),
            histogram.numpy().tolist(),
            durations.numpy(),
        )

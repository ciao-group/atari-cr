from enum import Enum
from typing import List, Optional, OrderedDict, Tuple, TypedDict, Union

import cv2
import numpy as np
import torch

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

class RecordBuffer(TypedDict):
    """
    Buffer saving the env history for one episode
    """
    rgb: List[np.ndarray]
    state: List[np.ndarray]
    action: List[np.ndarray]
    reward: List[np.ndarray]
    raw_reward: List[np.ndarray]
    done: List[np.ndarray]
    truncated: List[np.ndarray]
    info: List[np.ndarray]
    return_reward: List[np.ndarray]
    episode_pauses: List[int]
    fov_loc: List[np.ndarray]
    fov_size: Tuple[int, int]

    @staticmethod
    def new():
        buffer: RecordBuffer = { key: [] for key in RecordBuffer.__annotations__ }
        buffer["fov_size"] = (0, 0)
        return buffer

    @staticmethod
    def from_file(file_path: str, video_path: Optional[str] = None):
        record_buffer = torch.load(file_path, weights_only=False)
        video = cv2.VideoCapture(video_path or record_buffer["rgb"])
        frames = []
        while True:
            ret, frame = video.read()
            if not ret: break
            frames.append(frame)

        record_buffer["rgb"] = frames
        record_buffer: RecordBuffer
        return record_buffer

    # def to_atari_head(self, game: str):
    #     """
    #     Create a GazeDataset from the buffer.

    #     :param str game: The atari game recorded in the buffer. Required to know which
    #         action is the pause action
    #     """
    #     from atari_cr.atari_head import GazeDataset

    #     ACTION_COUNT = {
    #         "ms_pacman": 8,
    #         "boxing": 17,
    #         "breakout": 3,
    #         "road_runner": 17
    #     }
    #     assert game in ACTION_COUNT.keys(), f"Game {game} not supported"

    #     gazes = []
    #     for action in self["action"]:
    #         # Check for pause action
    #         if action["motor_action"] == ACTION_COUNT[game]:
    #             gazes.append(action["sensory_action"].copy())
    #             action = None
    #         else:
    #             gazes.append(action["sensory_action"])
    #             action["sensory_action"] = gazes.copy()
    #             gazes = []

    #     atari_head_data = {
    #         "frames": [],
    #         "sensory_actions": [],
    #     }
    #     for frame, action in zip(self["rgb"], self["action"]):
    #         if action is not None:
    #             atari_head_data["frames"].append(frame)
    #             atari_head_data["sensory_actions"].append(action["sensory_action"])

    #     return GazeDataset.from_gaze_data(*atari_head_data.values())

class RecordBufferSample(TypedDict):
    """
    Buffer saving one environment step

    :var Array[256,256,3] rgb: RGB game image of type uint8
    :var state ? ?: Observation masked by the fovea of the agent
    :var OrderedDict action: Dict containing int64 `motor_action` and Array[2]
        `sensory_action`
    :var np.int64 reward: Reward received from the transition to this timestep
    :var np.int64 raw_reward: Raw reward without pause costs
    :var bool done: Whether this is the last episode timestep
    :var bool truncated: Whether the episode is truncated after this timestep
    :var StepInfo info: Dict containing addtional information on this timestep
    :var np.int64 return_reward: ?
    :var int pauses: Number of pauses made during this timestep
    :var List[Array[2]] fov_loc: Gazes made during this timestep
    """
    rgb: np.ndarray
    # fov_state: np.ndarray
    action: OrderedDict[str, Union[np.int64, np.ndarray]]
    reward: np.int64
    raw_reward: np.int64
    done: bool
    truncated: bool
    info: StepInfo
    return_reward: np.int64
    pauses: int
    fov_loc: List[np.ndarray]

from enum import Enum
from typing import List, Tuple, TypedDict

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
        buffer = dict((key, []) for key in RecordBuffer.__annotations__)
        buffer["fov_size"] = (0, 0)
        return buffer     

    @staticmethod
    def from_file(file_path: str):
        record_buffer = torch.load(file_path)
        video = cv2.VideoCapture(record_buffer["rgb"])
        frames = []
        while True:
            ret, frame = video.read()
            if not ret: break
            frames.append(frame)

        record_buffer["rgb"] = frames
        record_buffer: RecordBuffer
        return record_buffer

    def to_atari_head(self, game: str): 
        """
        Create a GazeDataset from the buffer.

        :param str game: The atari game recorded in the buffer. Required to know which action is the pause action
        """
        from atari_cr.atari_head import GazeDataset

        ACTION_COUNT = {
            "ms_pacman": 8,
            "boxing": 17,
            "breakout": 3,
            "road_runner": 17
        }
        assert game in ACTION_COUNT.keys(), f"Game {game} not supported"

        gazes = []
        for action in self["action"]:
            # Check for pause action
            if action["motor_action"] == ACTION_COUNT[game]:
                gazes.append(action["sensory_action"].copy())
                action = None
            else:
                gazes.append(action["sensory_action"])
                action["sensory_action"] = gazes.copy()
                gazes = []

        atari_head_data = {
            "frames": [],
            "sensory_actions": [],
        }
        for frame, action in zip(self["rgb"], self["action"]):
            if action is not None:
                atari_head_data["frames"].append(frame)
                atari_head_data["sensory_actions"].append(action["sensory_action"])

        return GazeDataset.from_gaze_data(*atari_head_data.values())
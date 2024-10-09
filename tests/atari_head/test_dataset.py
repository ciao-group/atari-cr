import unittest

from atari_cr.atari_head.dataset import GazeDataset

@unittest.skip("WIP")
class TestFromGameData(unittest.TestCase):
    def test_main(self):
        video_files = [
            "tests/assets/ms_pacman_130_1m_recordings/seed0_step1000000_eval00.mp4",
            "tests/assets/ms_pacman_130_1m_recordings/seed0_step1000000_eval04.mp4"
        ]
        metadata_files = [
            "tests/assets/ms_pacman_130_1m_recordings/seed0_step1000000_eval00.pt",
            "tests/assets/ms_pacman_130_1m_recordings/seed0_step1000000_eval04.pt"
        ]
        dataset = GazeDataset.from_game_data(video_files, metadata_files)

if __name__ == '__main__':
    unittest.main()

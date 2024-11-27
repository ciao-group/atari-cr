import os
import unittest

from atari_cr.atari_head.dataset import GazeDataset
from atari_cr.models import EpisodeRecord

@unittest.skip
class TestFromGameData(unittest.TestCase):
    def test_main(self):
        episode_dirs = [
        "tests/assets/ray-10-16/" + dir for dir in os.listdir("tests/assets/ray-10-16")]
        dataset = GazeDataset.from_game_data(
            [EpisodeRecord.load(dir) for dir in episode_dirs])
        self.assertEqual(len(dataset.frames), len(dataset.saliency))
        self.assertEqual(len(dataset.frames),
            len(dataset.train_indices) + len(dataset.val_indices) \
                + len(episode_dirs) * 3)

if __name__ == '__main__':
    unittest.main()

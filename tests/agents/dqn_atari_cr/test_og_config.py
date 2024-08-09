import unittest
import argparse
from sys import argv

from atari_cr.agents.dqn_atari_cr.main import main, parse_args

class MainTestCase(unittest.TestCase):
    """ Deterministic test for OG config """

    def test_main(self):
        # TODO: Implement no logging whatsoever
        argv.extend([
            "--clip-reward",
            "--capture-video",
            "--exp-name", "test_og",
            "--total-timesteps", "1000000"
        ])

        args = parse_args()
        eval_returns = main(args)

        self.assertEqual(eval_returns, [1, 2, 3])

if __name__ == '__main__':
    unittest.main()
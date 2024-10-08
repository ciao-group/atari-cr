import unittest
from gymnasium import spaces

from atari_cr.module_overrides import get_action_dim

class TestGetActionDim(unittest.TestCase):
    def test_box_space(self):
        action_space = spaces.Box(low=0, high=1, shape=(3, 4))
        expected_dim = 12
        self.assertEqual(get_action_dim(action_space), expected_dim)

        action_space = spaces.Box(low=0, high=1, shape=(5,))
        expected_dim = 5
        self.assertEqual(get_action_dim(action_space), expected_dim)

    def test_discrete_space(self):
        action_space = spaces.Discrete(5)
        expected_dim = 1
        self.assertEqual(get_action_dim(action_space), expected_dim)

    def test_multidiscrete_space(self):
        action_space = spaces.MultiDiscrete([2, 3, 4])
        expected_dim = 3
        self.assertEqual(get_action_dim(action_space), expected_dim)

    def test_multibinary_space(self):
        action_space = spaces.MultiBinary(10)
        expected_dim = 10
        self.assertEqual(get_action_dim(action_space), expected_dim)

        action_space = spaces.MultiBinary([4, 3])
        with self.assertRaises(AssertionError):
            get_action_dim(action_space)

    def test_dict_space(self):
        action_space = spaces.Dict({
            'position': spaces.Box(low=0, high=1, shape=(3,)),
            'velocity': spaces.Discrete(2)
        })
        expected_dim = [3, 1]
        self.assertEqual(get_action_dim(action_space), expected_dim)

        action_space = spaces.Dict({
            'action1': spaces.MultiDiscrete([2, 3]),
            'action2': spaces.MultiBinary(5),
            'action3': spaces.Box(low=0, high=1, shape=(4,))
        })
        expected_dim = [2, 5, 4]
        self.assertEqual(get_action_dim(action_space), expected_dim)

    def test_unsupported_space(self):
        action_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(3)))
        with self.assertRaises(NotImplementedError):
            get_action_dim(action_space)

if __name__ == '__main__':
    unittest.main()

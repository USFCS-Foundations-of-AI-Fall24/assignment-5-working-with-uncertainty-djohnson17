import unittest

from HMM import HMM


class TestHMM(unittest.TestCase):
    def test_load(self):
        hmm = HMM()
        hmm.load('cat')

        expected_transitions = {
            '#': {'happy': 0.5, 'grumpy': 0.5, 'hungry': 0.0},
            'happy': {'happy': 0.5, 'grumpy': 0.1, 'hungry':0.4},
            'grumpy': {'happy': 0.6, 'grumpy': 0.3, 'hungry': 0.1},
            'hungry': {'happy': 0.1, 'grumpy': 0.6, 'hungry': 0.3}
        }

        expected_emissions = {
            'happy': {'meow': 0.3, 'purr': 0.5, 'silent': 0.2},
            'grumpy': {'meow': 0.4, 'purr': 0.1, 'silent': 0.5},
            'hungry': {'meow': 0.6, 'purr': 0.2, 'silent': 0.2}
        }

        self.assertEqual(hmm.transitions, expected_transitions)
        self.assertEqual(hmm.emissions, expected_emissions)


if __name__ == '__main__':
    unittest.main()

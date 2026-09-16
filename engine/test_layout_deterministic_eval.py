import unittest
from .layout_deterministic_eval import validate_workspace, validate_repeatability


class DeterministicProtocolTests(unittest.TestCase):
    def test_repeatability_fails_closed(self):
        runs = [{'outputHash': 'x', 'finalStateHash': 'y'},
                {'outputHash': 'x', 'finalStateHash': 'y', 'versusFirst': {'bitwiseEqual': True}},
                {'outputHash': 'x', 'finalStateHash': 'y', 'versusFirst': {'bitwiseEqual': True}}]
        proof = {'schema': 'frozen-forward-repeatability-v1', 'hashes': {'candidate': 'z'},
                 'parametersUnchanged': True,
                 'results': [{'strictDeterministicAlgorithms': True, 'finished': True, 'runs': runs}]}
        validate_repeatability(proof, 'z')
        runs[2]['finalStateHash'] = 'different'
        with self.assertRaises(ValueError):
            validate_repeatability(proof, 'z')

    def test_config(self):
        validate_workspace(':4096:8')
        validate_workspace(':16:8')
        for value in (None, '', '4096:8', ':0:0'):
            with self.assertRaises(ValueError):
                validate_workspace(value)


if __name__ == '__main__':
    unittest.main()

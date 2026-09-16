"""CPU fixture tests only; no dataset, checkpoint or live service required."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
MODULES = [
    'test_deployment',
    'test_swarm', 'test_continuous_view', 'test_service_training',
    'test_supervised_steering', 'test_supervised_joint',
    'test_fast_frozen_policy', 'test_neural_telemetry', 'test_phase_viewer',
    'test_fixed_layout_curriculum', 'test_fixed_layout_review',
    'test_layout_recovery', 'test_layout_recovery_protocol',
    'test_layout_service_metrics', 'test_layout_deterministic_eval',
]
if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromNames(['engine.'+m for m in MODULES])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())

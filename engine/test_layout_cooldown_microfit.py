import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import ExitStack
import numpy as np
import torch
from .layout_cooldown_microfit import load_revised_labels, main
from .layout_cooldown_labels import VERSION
from .layout_recovery_demonstrations import file_hash
from .test_layout_cargo_response_probe import CargoBrain


def fixture(folder, original):
    folder.mkdir()
    prior = original.numpy()
    revised = prior.copy()
    revised[81:85, 1, 2] = 2.2
    with (folder/'labels.npz').open('wb') as stream:
        np.savez_compressed(stream, original=prior, revised=revised)
    proof = {'selectedWindows': [{'episode': 0}], 'physicalTrainingData': {'datasetManifestHash': 'fixture'}}
    record = {'version': VERSION, 'finished': True, 'physicalTrajectoriesUnchanged': True,
              'teacherAtInference': False, 'newRuntimeController': False, 'isServiceEvidence': False,
              'extraFrames': 4, 'extraSeconds': .2, 'proofSelections': proof['selectedWindows'],
              'datasetManifestHash': 'fixture', 'audits': [{'episode': 0, 'allChangesIgnoredByActualCooldown': True,
                    'everyObservationBodyAndEventExactlyMatched': True, 'isServiceEvidence': False}],
              'originalPhysicsSourceHashes': {'layout_recovery_world.py': file_hash(Path(__file__).with_name('layout_recovery_world.py'))},
              'sourceHash': file_hash(Path(__file__).with_name('layout_cooldown_labels.py')),
              'labelsFileHash': file_hash(folder/'labels.npz')}
    (folder/'manifest.json').write_text(json.dumps(record))
    return proof, record


class CooldownMicrofitTests(unittest.TestCase):
    def test_loader_preserves_original_and_rejects_changed_attestation_or_inputs(self):
        original = torch.full((96, 4, 3), .2)
        original[80, 1, 2] = 2.2
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)/'labels'
            proof, record = fixture(folder, original)
            target, _ = load_revised_labels(folder, proof, original, 'cpu')
            self.assertEqual(int((target[..., 2] > 1.).sum()), 5)
            self.assertEqual(int((original[..., 2] > 1.).sum()), 1)
            altered = original.clone()
            altered[1, 1, 0] += .1
            with self.assertRaises(ValueError):
                load_revised_labels(folder, proof, altered, 'cpu')
            for change in ({'teacherAtInference': True}, {'labelsFileHash': 'bad'}, {'audits': []},
                           {'sourceHash': 'bad'}, {'extraFrames': 20}):
                (folder/'manifest.json').write_text(json.dumps({**record, **change}))
                with self.assertRaises(ValueError):
                    load_revised_labels(folder, proof, original, 'cpu')

    def test_actual_matched_trainer_updates_and_saves_with_honest_metric_labels(self):
        model = CargoBrain()
        model.interface = 'fixture'
        model.fingerprint = lambda: model.fixed_hash
        before = model.checkpoint_hash()
        x = torch.ones((96, 4, 297))
        x[:, ::2, 126] = 0.
        original = torch.full((96, 4, 3), .2)
        original[80, 1, 2] = 2.2
        state = torch.zeros((4, 4))
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            folder = Path(temporary)
            proof, _ = fixture(folder/'labels', original)
            parent = folder/'parent.npz'
            parent.write_bytes(b'unit-test-only-parent')
            cache = folder/'cache'
            cache.mkdir()
            (cache/'manifest.json').write_text('{}')
            proof.update(parameterHash=before, candidateFileHash=file_hash(parent),
                         cacheManifestHash=file_hash(cache/'manifest.json'))
            proof_path = folder/'proof.json'
            proof_path.write_text(json.dumps(proof))
            args = SimpleNamespace(root=folder, candidate=parent, cache=cache, proof=proof_path,
                                   labels=folder/'labels', out=folder/'out', updates=2,
                                   save_every=1, lr=.003, tonic_lr=.00001)
            prefix = 'engine.layout_cooldown_microfit.'
            stack.enter_context(patch(prefix+'load_model', return_value=model))
            stack.enter_context(patch(prefix+'DemonstrationCache', return_value=object()))
            stack.enter_context(patch(prefix+'attested_batch', return_value=(x, original, state)))
            stack.enter_context(patch(prefix+'save_model', side_effect=lambda path, brain: path.write_bytes(b'fixture')))
            stack.enter_context(patch('builtins.print'))
            main(args)
            self.assertNotEqual(before, model.checkpoint_hash())
            status = json.loads((args.out/'status.json').read_text())
            self.assertTrue(status['finished'])
            self.assertTrue(status['gripMetricsUseRevisedTrainingLabels'])
            manifest = json.loads((args.out/'manifest.json').read_text())
            self.assertFalse(manifest['teacherAtInference'])
            self.assertTrue(manifest['optimizerFreshByDesign'])
            self.assertFalse(manifest['remoteExperienceUsed'])
            self.assertTrue((args.out/'candidate-2.npz').exists())
            self.assertTrue((args.out/'optimizer-2.pt').exists())
            self.assertEqual(file_hash(parent), proof['candidateFileHash'])
            self.assertEqual(int((original[..., 2] > 1.).sum()), 1)


if __name__ == '__main__':
    unittest.main()

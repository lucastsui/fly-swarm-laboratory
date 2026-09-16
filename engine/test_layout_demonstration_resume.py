import copy
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from .layout_demonstration_resume import prepare_continuation, restore_continuation_optimizer
from .layout_demonstration_train import DemonstrationCache, main, validate_bounds
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_world import INTERFACE
from .test_layout_demonstration_train import tiny_cache
from .test_layout_microfit_flow import TinyTrainBrain


class AuditedTinyBrain(TinyTrainBrain):
    def audit(self, initial):
        return {**super().audit(initial), 'signsPreserved': True}


def test_cache(path, model, candidate):
    metadata = tiny_cache(path, model)
    metadata.update(candidateFileHash=file_hash(candidate), datasetFiles=[{'file': 'fixture', 'sha256': 'same-data'}])
    (path/'manifest.json').write_text(json.dumps(metadata))


class DemonstrationResumeTests(unittest.TestCase):
    def test_actual_resume_uses_global_steps_fresh_cache_and_retains_source(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            folder = Path(temp)
            model = AuditedTinyBrain()
            model.interface = INTERFACE
            parent = folder/'parent.npz'
            parent.write_bytes(b'initial-test-parent')
            token = folder/'token'
            token.write_text('test-only-token')
            test_cache(folder/'cache', model, parent)
            args = SimpleNamespace(root=folder, candidate=parent, cache=folder/'cache', out=folder/'source',
                                   token_file=token, cert=folder/'cert', key=folder/'key', updates=4,
                                   burn=2, gradient_frames=2, save_every=2, lr=.0003, tonic_lr=.000001,
                                   seed=9380001, recent_corrections=True, require_stratified_corrections=True)
            prefix = 'engine.layout_demonstration_train.'
            stack.enter_context(patch(prefix+'load_model', return_value=model))
            stack.enter_context(patch(prefix+'save_model', side_effect=lambda path, brain:
                                      np.savez_compressed(path, gains=brain.log_gains.detach().numpy(),
                                                          tonic=brain.tonic.detach().numpy())))
            stack.enter_context(patch(prefix+'start_server', return_value=SimpleNamespace(shutdown=lambda: None)))
            stack.enter_context(patch(prefix+'Exchange.pop', return_value=None))
            stack.enter_context(patch(prefix+'torch.cuda.get_device_name', return_value='CPU unit test'))
            stack.enter_context(patch(prefix+'torch.cuda.max_memory_allocated', return_value=0))
            stack.enter_context(patch(prefix+'time.sleep'))
            log = []
            stack.enter_context(patch('builtins.print', side_effect=lambda *items, **kw:
                                      log.append(' '.join(map(str, items)))))
            main(args)
            source_log = folder/'source.log'
            source_log.write_text('\n'.join(log)+'\n')
            retained = {p: p.read_bytes() for p in args.out.iterdir() if p.is_file()}
            candidate = args.out/'candidate-4.npz'
            test_cache(folder/'fresh-cache', model, candidate)
            next_args = SimpleNamespace(**{**vars(args), 'candidate': candidate, 'cache': folder/'fresh-cache',
                                            'out': folder/'next', 'resume_demonstration_run': args.out})
            bank = DemonstrationCache(next_args.cache, model.checkpoint_hash(), model.fixed_hash, 2, 2, 4)
            continuation = prepare_continuation(next_args, model, bank)
            self.assertEqual(continuation['start'], 4)
            self.assertFalse(continuation['record']['physicalWorldsResumed'])
            for changes in ({'lr': .1}, {'recent_corrections': False}, {'require_stratified_corrections': False},
                            {'cache': args.cache}, {'candidate': args.out/'candidate-2.npz'}):
                bad = SimpleNamespace(**{**vars(next_args), **changes})
                with self.assertRaises(ValueError):
                    prepare_continuation(bad, model, bank)
            for name, change in [('status.json', {'finished': False}), ('result.json', {'updates': 5}),
                                 ('manifest.json', {'sourceHashes': {}})]:
                path = args.out/name
                path.write_text(json.dumps({**json.loads(retained[path]), **change}))
                try:
                    with self.assertRaises(ValueError):
                        prepare_continuation(next_args, model, bank)
                finally:
                    path.write_bytes(retained[path])
            original_fresh = copy.deepcopy(bank.manifest)
            for change in ({'datasetManifestHash': 'changed'}, {'windows': []},
                           {'parameterHash': 'old-parent'}, {'candidateFileHash': 'wrong'}):
                bank.manifest = {**original_fresh, **change}
                with self.assertRaises(ValueError):
                    prepare_continuation(next_args, model, bank)
            main(next_args)
            history = json.loads((next_args.out/'history.json').read_text())
            self.assertEqual([r['update'] for r in history], [5, 6, 7, 8])
            self.assertTrue(all(r['sourceVersion'] == 4 for r in history))
            result = json.loads((next_args.out/'result.json').read_text())
            self.assertEqual((result['startUpdate'], result['updates'], result['updatesThisRun']), (4, 8, 4))
            manifest = json.loads((next_args.out/'manifest.json').read_text())
            self.assertEqual((manifest['prefixParentUpdate'], manifest['maximumPrefixAgeUpdates']), (4, 80))
            self.assertTrue((next_args.out/'candidate-4.npz').exists())
            self.assertTrue((next_args.out/'candidate-8.npz').exists())
            state = torch.load(next_args.out/'optimizer-8.pt', weights_only=True)
            self.assertTrue(all(float(s['step']) == 8 for s in state['optimizer']['state'].values()))
            for path, contents in retained.items():
                self.assertEqual(path.read_bytes(), contents)

            # Select an EARLIER saved update from the same finished source.
            early = args.out/'candidate-2.npz'
            with np.load(early, allow_pickle=False) as saved, torch.no_grad():
                model.log_gains.copy_(torch.tensor(saved['gains']))
                model.tonic.copy_(torch.tensor(saved['tonic']))
            test_cache(folder/'earlier-cache', model, early)
            selected = SimpleNamespace(**{**vars(args), 'candidate': early,
                                          'cache': folder/'earlier-cache', 'out': folder/'selected',
                                          'resume_demonstration_run': args.out,
                                          'resume_checkpoint_update': 2, 'updates': 2})
            bank = DemonstrationCache(selected.cache, model.checkpoint_hash(), model.fixed_hash, 2, 2, 4)
            proof = prepare_continuation(selected, model, bank)
            self.assertEqual((proof['start'], proof['record']['sourceFinalUpdate']), (2, 4))
            self.assertTrue(proof['record']['selectedIntermediateCheckpoint'])
            self.assertTrue(proof['record']['selectedCheckpointAudit']['finite'])
            self.assertIn('source.log', proof['record']['sourceFileSHA256'])
            for update in (-1, 0, 1, 3, 5, True):
                with self.assertRaises(ValueError):
                    prepare_continuation(SimpleNamespace(**{**vars(selected), 'resume_checkpoint_update': update}),
                                         model, bank)
            with self.assertRaises(ValueError):
                validate_bounds(SimpleNamespace(**{**vars(selected), 'resume_demonstration_run': None}))
            original_log = source_log.read_text()
            lines = original_log.splitlines()
            original_publication = next(json.loads(line[len('CHECKPOINT '):]) for line in lines
                                        if line.startswith('CHECKPOINT ') and json.loads(line[len('CHECKPOINT '):])['version'] == 2)
            for mutation in ('missing', 'duplicate', 'run', 'hash', 'file', 'version', 'interface'):
                changed = copy.deepcopy(original_publication)
                if mutation == 'run': changed['runId'] = 'other-run'
                elif mutation == 'hash': changed['checkpoint']['sha256'] = 'wrong'
                elif mutation == 'file': changed['checkpoint']['file'] = 'candidate-4.npz'
                elif mutation == 'version': changed['update'] = 3
                elif mutation == 'interface': changed['interface'] = 'other-interface'
                others = [line for line in lines if not line.startswith('CHECKPOINT ') or
                          json.loads(line[len('CHECKPOINT '):])['version'] != 2]
                count = 0 if mutation == 'missing' else 2 if mutation == 'duplicate' else 1
                source_log.write_text('\n'.join(others+['CHECKPOINT '+json.dumps(changed)]*count))
                with self.assertRaises(ValueError):
                    prepare_continuation(selected, model, bank)
            source_log.write_text(original_log)
            fit_path = args.out/'demonstration-fit-2.json'
            fit = fit_path.read_bytes()
            fit_path.write_text(json.dumps({**json.loads(fit), 'checkpointHash': 'other'}))
            with self.assertRaises(ValueError):
                prepare_continuation(selected, model, bank)
            fit_path.write_bytes(fit)
            with patch.object(model, 'audit', return_value={'finite': False}):
                with self.assertRaises(ValueError):
                    prepare_continuation(selected, model, bank)
            with torch.no_grad():
                initial_tonic = model.tonic.clone()
                model.tonic.fill_(.11)
            with self.assertRaisesRegex(ValueError, 'bounds'):
                prepare_continuation(selected, model, bank)
            with torch.no_grad():
                model.tonic.copy_(initial_tonic)
            main(selected)
            selected_manifest = json.loads((selected.out/'manifest.json').read_text())
            self.assertEqual((selected_manifest['startUpdate'], selected_manifest['finalUpdate']), (2, 4))
            self.assertEqual(selected_manifest['continuation']['sourceFinalUpdate'], 4)
            self.assertEqual([r['update'] for r in json.loads((selected.out/'history.json').read_text())], [3, 4])
            selected_optimizer = torch.load(selected.out/'optimizer-4.pt', weights_only=True)
            self.assertTrue(all(float(s['step']) == 4 for s in selected_optimizer['optimizer']['state'].values()))
            # Same tiny inputs/cached states and Adam+sampler restore reproduce
            # the uninterrupted source's next two updates exactly.
            with np.load(args.out/'candidate-4.npz', allow_pickle=False) as original:
                np.testing.assert_array_equal(model.log_gains.detach().numpy(), original['gains'])
                np.testing.assert_array_equal(model.tonic.detach().numpy(), original['tonic'])
            for path, contents in retained.items():
                self.assertEqual(path.read_bytes(), contents)
            self.assertEqual(source_log.read_text(), original_log)

            # Explicitly expand real physical TRAINING data without resetting Adam.
            from .test_layout_demonstration_curriculum import physical_curriculum_fixture
            dataset, cache = physical_curriculum_fixture(folder, model, candidate)
            expanded = SimpleNamespace(**{**vars(next_args), 'cache': cache, 'out': folder/'expanded',
                                          'updates': 2, 'resume_new_demonstration_dataset': dataset})
            expanded_bank = DemonstrationCache(cache, model.checkpoint_hash(), model.fixed_hash, 2, 2, 4)
            with self.assertRaises(ValueError):
                prepare_continuation(SimpleNamespace(**{**vars(expanded),
                                     'resume_new_demonstration_dataset': None}), model, expanded_bank)
            main(expanded)
            expanded_manifest = json.loads((expanded.out/'manifest.json').read_text())
            record = expanded_manifest['continuation']
            self.assertFalse(record['sameDemonstrationDataset'])
            self.assertTrue(record['optimizerMomentsPreserved'])
            self.assertTrue(record['demonstrationDatasetChange']['allCachedInputsAndTargetsMatched'])
            self.assertEqual(record['demonstrationDatasetChange']['episodesPerFamily'],
                             dict.fromkeys(('compact', 'wide', 'rotated', 'permuted'), 2))
            expanded_optimizer = torch.load(expanded.out/'optimizer-6.pt', weights_only=True)
            self.assertTrue(all(float(s['step']) == 6 for s in expanded_optimizer['optimizer']['state'].values()))
            for path, contents in retained.items():
                self.assertEqual(path.read_bytes(), contents)

            # Explicit step-size branch keeps the actual saved Adam state/base
            # controls, logs the revision, and inherits it on later continuation.
            with np.load(candidate, allow_pickle=False) as saved, torch.no_grad():
                model.log_gains.copy_(torch.tensor(saved['gains']))
                model.tonic.copy_(torch.tensor(saved['tonic']))
            focused = SimpleNamespace(**{**vars(expanded), 'out': folder/'focused',
                                         'event_focused_demonstrations': dataset})
            main(focused)
            fm = json.loads((focused.out/'manifest.json').read_text())
            self.assertEqual(fm['eventFocusedDataset'], str(dataset.resolve()))
            self.assertEqual(fm['eventFocusedSampling']['actorsPerBatch'], 16)
            self.assertTrue(fm['eventFocusedSampling']['allCachedInputsAndTargetsMatched'])
            self.assertTrue(fm['demonstrationSamplingRevision']['samplingPolicyChanged'])
            self.assertTrue(fm['continuation']['optimizerMomentsPreserved'])
            self.assertEqual(fm['effectiveStepScale'], 1.)
            fh = json.loads((focused.out/'history.json').read_text())
            self.assertTrue(all(len(r['sourceDetail']['selections']) == 16 for r in fh))
            self.assertTrue(all(r['sourceDetail']['noNewBrainInputs'] for r in fh))
            for path, contents in retained.items():
                self.assertEqual(path.read_bytes(), contents)
            with np.load(candidate, allow_pickle=False) as saved, torch.no_grad():
                model.log_gains.copy_(torch.tensor(saved['gains']))
                model.tonic.copy_(torch.tensor(saved['tonic']))
            scaled = SimpleNamespace(**{**vars(next_args), 'out': folder/'scaled',
                                        'updates': 2, 'revise_step_scale': 4.})
            main(scaled)
            sm = json.loads((scaled.out/'manifest.json').read_text())
            self.assertEqual(sm['effectiveStepScale'], 4.)
            self.assertTrue(sm['updateScaleExperiment']['changed'])
            self.assertTrue(sm['continuation']['optimizerMomentsPreserved'])
            self.assertEqual(sm['effectiveLearningRates'], [.0012, .000004])
            so = torch.load(scaled.out/'optimizer-6.pt', weights_only=True)['optimizer']
            self.assertEqual([g['lr'] for g in so['param_groups']], [.0003, .000001])
            self.assertTrue(all(float(s['step']) == 6 for s in so['state'].values()))
            self.assertTrue(all(r['effectiveStepScale'] == 4. for r in
                                json.loads((scaled.out/'history.json').read_text())))
            scaled_candidate = scaled.out/'candidate-6.npz'
            test_cache(folder/'scaled-cache', model, scaled_candidate)
            inherited = SimpleNamespace(**{**vars(next_args), 'out': folder/'inherited',
                                            'resume_demonstration_run': scaled.out,
                                            'candidate': scaled_candidate, 'cache': folder/'scaled-cache', 'updates': 2})
            main(inherited)
            inherited_manifest = json.loads((inherited.out/'manifest.json').read_text())
            self.assertEqual(inherited_manifest['effectiveStepScale'], 4.)
            self.assertFalse(inherited_manifest['updateScaleExperiment']['changed'])
            self.assertTrue(all(float(s['step']) == 8 for s in
                torch.load(inherited.out/'optimizer-8.pt', weights_only=True)['optimizer']['state'].values()))
            for path, contents in retained.items():
                self.assertEqual(path.read_bytes(), contents)

    def test_preserved_adam_and_rng_match_uninterrupted_next_step(self):
        def create():
            model = torch.nn.Linear(2, 1, dtype=torch.float64)
            opt = torch.optim.Adam([{'params': [model.weight], 'lr': .003, 'eps': 1e-14},
                                    {'params': [model.bias], 'lr': .00001, 'eps': 1e-10}])
            return model, opt

        def step(model, opt):
            opt.zero_grad(set_to_none=True)
            model(torch.ones((3, 2), dtype=torch.float64)).square().mean().backward()
            opt.step()

        model, optimizer = create()
        for _ in range(3):
            step(model, optimizer)
        next_model, next_optimizer = create()
        next_model.load_state_dict(model.state_dict())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'optimizer-3.pt'
            rng = np.random.default_rng(9380001)
            rng.integers(100, size=20)
            payload = {'updates': 3, 'rng': rng.bit_generator.state, 'optimizer': optimizer.state_dict()}
            torch.save(payload, path)
            continuation = {'start': 3, 'optimizerPath': path,
                            'record': {'sourceFileSHA256': {path.name: file_hash(path)}}}
            restored_rng = np.random.default_rng()
            restored_rng.bit_generator.state = restore_continuation_optimizer(next_optimizer, continuation)
            np.testing.assert_array_equal(rng.integers(100, size=30), restored_rng.integers(100, size=30))
            step(model, optimizer)
            step(next_model, next_optimizer)
            for a, b in zip(model.parameters(), next_model.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            for mutation in ('step', 'lr', 'beta', 'shape', 'nan', 'negative', 'checkpoint'):
                bad = copy.deepcopy(payload)
                first = bad['optimizer']['state'][0]
                if mutation == 'step': first['step'] = torch.tensor(4.)
                elif mutation == 'lr': bad['optimizer']['param_groups'][0]['lr'] = .1
                elif mutation == 'beta': bad['optimizer']['param_groups'][0]['betas'] = (.8, .99)
                elif mutation == 'shape': first['exp_avg'] = torch.zeros(3, dtype=torch.float64)
                elif mutation == 'nan': first['exp_avg'].fill_(float('nan'))
                elif mutation == 'negative': first['exp_avg_sq'].fill_(-1.)
                else: bad['updates'] = 4
                torch.save(bad, path)
                continuation['record']['sourceFileSHA256'][path.name] = file_hash(path)
                _, fresh = create()
                with self.assertRaises(ValueError):
                    restore_continuation_optimizer(fresh, continuation)
                self.assertFalse(fresh.state)
            continuation['record']['sourceFileSHA256'][path.name] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'file changed'):
                restore_continuation_optimizer(next_optimizer, continuation)


if __name__ == '__main__':
    unittest.main()

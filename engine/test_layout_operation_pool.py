import copy
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from .layout_operation_pool import OperationPool, pool_metadata
from .layout_operation_prefix import KINDS, CONTEXTS


def manifest(seed, digest):
    files = [{'seed': seed+i, 'kind': k} for i,k in enumerate(KINDS)]
    return {'parameterHash': 'p', 'fixedHash': 'f', 'behaviorParameterHash': 'p',
            'behaviorCanonicalRun': 'run', 'behaviorCanonicalVersion': 1,
            'prefixCanonicalRun': 'run', 'prefixCanonicalVersion': 1,
            'candidateFileHash': 'file', 'interface': 'i', 'physics': 'physical',
            'labelVersion': 'labels', 'burn': 64, 'lossFrames': 32,
            'datasetManifestHash': digest, 'datasetFiles': files,
            'windows': [{'episode': i, 'seed': seed+i, 'kind': k, 'category': c,
                         'start': 0, 'stop': 96, 'lossStart': 64, 'focusTick': 80, 'fly': 0}
                        for i,k in enumerate(KINDS) for c in CONTEXTS]}


class PoolTests(unittest.TestCase):
    def test_original_metadata_preserved_with_explicit_mapping(self):
        a, b = manifest(100, 'a'), manifest(200, 'b')
        originals = copy.deepcopy([a,b]); pooled = pool_metadata([a,b])
        self.assertEqual([a,b], originals)
        self.assertEqual(pooled['episodes'], 8)
        self.assertEqual(len(pooled['windows']), 32)
        self.assertEqual(pooled['windows'][16]['episode'], 4)
        self.assertEqual(pooled['windows'][16]['sourceEpisode'], 0)
        self.assertEqual(pooled['windows'][16]['sourceWindow'], 0)
        self.assertEqual(pooled['windows'][16]['sourceBank'], 1)
        self.assertEqual(pooled['windows'][16]['sourceDatasetManifestHash'], 'b')

    def test_wrong_identity_and_version_rejected(self):
        for key in ('parameterHash', 'fixedHash', 'behaviorParameterHash', 'behaviorCanonicalRun',
                    'behaviorCanonicalVersion', 'prefixCanonicalRun', 'prefixCanonicalVersion',
                    'candidateFileHash', 'interface', 'physics', 'labelVersion', 'burn', 'lossFrames'):
            a, b = manifest(100, 'a'), manifest(200, 'b'); b[key] = 'wrong'
            with self.assertRaises(ValueError): pool_metadata([a,b])

    def test_duplicate_data_or_episodes_rejected(self):
        for b in (manifest(200, 'a'), manifest(100, 'b'), manifest(103, 'b')):
            with self.assertRaises(ValueError): pool_metadata([manifest(100,'a'), b])
        b = manifest(200, 'b'); b['datasetFiles'][1]['seed'] = 200
        with self.assertRaises(ValueError): pool_metadata([manifest(100,'a'), b])

    def test_wrong_original_mapping_rejected(self):
        for key, value in (('episode', -1), ('episode', 4), ('kind', 'wrong'), ('seed', 201)):
            b = manifest(200, 'b'); b['windows'][0][key] = value
            with self.assertRaises(ValueError): pool_metadata([manifest(100,'a'), b])

    def test_pool_arrays_order_and_real_batch_indexing(self):
        class Bank:
            def __init__(self, m, fill):
                self.manifest = m
                for name, shape in {'x':(16,96,1,297), 'y':(16,96,1,3),
                                    'states':(16,3,1), 'motion':(16,32,1,2)}.items():
                    setattr(self,name,np.full(shape,fill,dtype=np.float32))
        a, b = Bank(manifest(100,'a'),1), Bank(manifest(200,'b'),2)
        with patch('engine.layout_operation_pool.OperationCache', side_effect=[a,b]), patch(
                'engine.layout_operation_pool.file_hash', return_value='hash'):
            pool = OperationPool([Path('a'),Path('b')],[Path('d1'),Path('d2')],'p','f',3)
        x,y,state,motion = pool.batch([16,0], 'cpu')
        self.assertEqual(tuple(x.shape), (96,2,297))
        self.assertTrue(np.all(x[:,0].numpy()==2) and np.all(x[:,1].numpy()==1))
        self.assertEqual(pool.groups[KINDS[0],CONTEXTS[0]], [0,16])
        self.assertEqual(pool.groups[KINDS[0],CONTEXTS[1]], [17,1])
        selected=[pool.manifest['windows'][pool.groups[k,c][0]]['sourceBank'] for k in KINDS for c in CONTEXTS]
        self.assertEqual(selected.count(0),8);self.assertEqual(selected.count(1),8)
        pool.x[0,0,0,0] = 9
        self.assertEqual(a.x[0,0,0,0], 1)


if __name__ == '__main__': unittest.main()

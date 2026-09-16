"""Explicit original-label learner corrections mixed with revised-label demos.

Selection is training-only. It does not select a sensory goal or alter any
observation, label, motor decoder or recurrent history.
"""
import json
from pathlib import Path
import numpy as np
import torch
from .layout_correction_prefix import VERSION, CONTEXTS, load_histories, correction_specs
from .layout_recovery_teacher import KINDS, LABEL_VERSION
from .layout_recovery_world import INTERFACE, PHYSICS, CHANNELS
from .layout_recovery_demonstrations import file_hash


class CorrectionCache:
    def __init__(self, folder, dataset, parameter_hash, fixed_hash, neurons=166700):
        folder,dataset=Path(folder),Path(dataset)
        m=json.loads((folder/'manifest.json').read_text())
        expected={'schema':VERSION,'finished':True,'parametersUnchanged':True,'parameterHash':parameter_hash,
                  'fixedHash':fixed_hash,'control':'learner-only','teacherActions':False,'optimizerUsed':False,
                  'labelVersion':LABEL_VERSION,'labelsUnchanged':True,'interface':INTERFACE,'physics':PHYSICS,
                  'isServiceEvidence':False,'burn':64,'gradientFrames':32}
        if any(m.get(k)!=v for k,v in expected.items()) or (folder/'failure.json').exists():
            raise ValueError('Incompatible or incomplete correction cache')
        required={'layout_correction_prefix.py','layout_batched_prefix_cache.py','layout_demonstration_cache.py'}
        if not required.issubset(m['sourceHashes']) or any(Path(n).name!=n or
                file_hash(Path(__file__).parent/n)!=h for n,h in m['sourceHashes'].items()):
            raise ValueError('Correction prefix source mismatch')
        episodes,data=load_histories(dataset,replay=False)
        if (file_hash(dataset/'manifest.json')!=m['datasetManifestHash'] or data['episodes']!=m['datasetFiles']
                or data['parameterHash']!=m['behaviorParameterHash']
                or data['canonicalRun']!=m['behaviorCanonicalRun']
                or data['canonicalVersion']!=m['behaviorCanonicalVersion']
                or data['fixedHash']!=fixed_hash or correction_specs(episodes)!=m['windows']
                or file_hash(folder/'cache.npz')!=m['cacheFileHash']):
            raise ValueError('Changed original correction history/specification')
        with np.load(folder/'cache.npz',allow_pickle=False) as z:
            if set(z.files)!={'observations','labels','states'}:
                raise ValueError('Unexpected cache arrays')
            self.x,self.y,self.states=[z[k].copy() for k in ('observations','labels','states')]
        n=len(m['windows'])
        for a,shape in zip((self.x,self.y,self.states),((n,96,1,CHANNELS),(n,96,1,3),(n,neurons,1))):
            if a.shape!=shape or a.dtype!=np.float32 or not np.isfinite(a).all():
                raise ValueError('Invalid correction prefix arrays')
        self.groups={(kind,context):[] for kind in KINDS for context in CONTEXTS}
        for i,s in enumerate(m['windows']):
            a=episodes[s['episode']][0]; sl=slice(s['start'],s['stop']); f=s['fly']
            if (not np.array_equal(self.x[i],a['observations'][sl,f:f+1])
                    or not np.array_equal(self.y[i],a['labels'][sl,f:f+1])):
                raise ValueError('Correction sensory/label window altered')
            self.groups[s['kind'],s['category']].append(i)
        if any(not ids for ids in self.groups.values()):
            raise ValueError('Missing correction context')
        self.manifest=m
        self.record={'source':'learner-only-correction','labelVersion':LABEL_VERSION,
                     'originalLabelsUnmodified':True,'datasetManifestHash':m['datasetManifestHash'],
                     'behaviorParameterHash':m['behaviorParameterHash'],'prefixParameterHash':parameter_hash,
                     'teacherActions':False,'noNewBrainInputs':True,'isServiceEvidence':False}

    def batch(self,indexes,device):
        return tuple(torch.as_tensor(np.concatenate([a[i] for i in indexes],axis=1),device=device)
                     for a in (self.x,self.y,self.states))

    def sample(self,rng,device):
        ids=[int(rng.choice(self.groups[k,c])) for k in KINDS for c in CONTEXTS]
        return (*self.batch(ids,device),{**self.record,'actors':16,
                   'selections':[{'windowIndex':i,**self.manifest['windows'][i]} for i in ids]})

    def diagnostic_indexes(self):
        return [self.groups[k,c][0] for k in KINDS for c in CONTEXTS]


class MixedCorrectionCurriculum:
    def __init__(self,demonstrations,corrections):
        if demonstrations.manifest['parameterHash']!=corrections.manifest['parameterHash']:
            raise ValueError('Both histories must use the same refreshed prefix parent')
        self.demonstrations,self.corrections=demonstrations,corrections
        self.manifest=demonstrations.manifest
        self.samples=self.correction_samples=0

    def sample(self,rng,device):
        self.samples+=1
        if self.samples%4==0:
            self.correction_samples+=1
            x,y,state,meta=self.corrections.sample(rng,device)
        else:
            x,y,state,meta=self.demonstrations.sample(rng,device)
            meta={**meta,'source':'service-balanced-demonstration'}
        return x,y,state,{**meta,'mixedSample':self.samples,'correctionSamples':self.correction_samples,
                         'teacherAtInference':False,'isServiceEvidence':False}

    def batch(self,indexes,device):
        return self.demonstrations.batch(indexes,device)

    def diagnostic_indexes(self):
        return self.demonstrations.diagnostic_indexes()

"""Full MaleCNS rate model with experimental three-factor synaptic plasticity.

No actor, learned decoder, backpropagation, or alternative motion controller.
Only existing fast afferents to MBONs, descending neurons and cb motor cells are
eligible. Modulation of that entire mask is an ENGINEERED volume-transmission
assumption, not a claim about reconstructed dopamine receptor distributions.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import numpy as np
import torch

SCHEMA = 'dopamine-haul-v1'
ALL_SYNAPSES_SCHEMA = 'dopamine-all-synapses-v2'
DT = .05


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def prepare(destination):
    from .plane import EmbodiedBrain, GRAPH, PROJECT
    b = EmbodiedBrain(device='cpu')
    with np.load(GRAPH/'graph.npz') as graph:
        crow, col = graph['crow'], graph['col']
        mask = (b.annotations.superclass.isin(['descending_neuron','cb_motor']) |
                b.annotations.type.fillna('').str.startswith('MBON')).to_numpy()
        rows = np.flatnonzero(mask)
        edges = np.concatenate([np.arange(crow[i],crow[i+1]) for i in rows])
        edges = edges[b.fast_mask[:,0].numpy()[col[edges]] > 0]
        post = np.repeat(np.arange(b.n),np.diff(crow))[edges]
        rows = np.unique(post)
        manifest = json.loads((PROJECT/'public/anatomy/manifest.json').read_text())
        lookup = {int(v):i for i,v in enumerate(b.ids)}
        nodes = [{'id':n['id'],'group':0 if 'sensory' in n['superclass'] else 2 if ('motor' in n['superclass'] or n['superclass']=='descending_neuron') else 1,
                  'position':n['soma'] or [0,0,0]} for n in manifest['neurons'] if n['id'] in lookup]
        destination.mkdir(parents=True,exist_ok=True)
        if not (destination/'graph.npz').exists():
            shutil.copy2(GRAPH/'graph.npz',destination/'graph.npz')
        np.savez(destination/'brain-spec.npz',plastic_edges=edges,pre=col[edges],post=post,plastic_rows=rows,
                 post_rank=np.searchsorted(rows,post),base=b.source_values[edges],
                 sensory_indices=b.sensory_indices.numpy(),sensory_channels=b.sensory_channels.numpy(),
                 tonic=b.tonic.numpy(),fast_mask=b.fast_mask.numpy(),
                 view_indices=np.array([lookup[n['id']] for n in nodes]),
                 **{f'motor_{k}':v.numpy() for k,v in b.motor_groups.items()},
                 **{f'dan_{k}':v.numpy() for k,v in b.dopamine_groups.items()})
        info = {'schema':SCHEMA,'neurons':b.n,'edges':len(col),'plasticSynapses':len(edges),'plasticCells':len(rows),
                'initialWeightHash':b.initial_hash,'plasticMaskHash':digest(edges),'sensory':len(b.sensory_indices),
                'nodes':nodes,'motorNeurons':{k:[int(b.ids[i]) for i in v] for k,v in b.motor_groups.items()}}
        (destination/'brain-info.json').write_text(json.dumps(info))
        print(json.dumps({k:v for k,v in info.items() if k not in ('nodes','motorNeurons')}))


class PlasticBrain:
    def __init__(self,root,batch=16,seed=1,all_synapses=False):
        self.root=Path(root); self.batch=batch; self.device=torch.device('cuda')
        self.info=json.loads((self.root/'brain-info.json').read_text())
        self.spec={k:v.copy() for k,v in np.load(self.root/'brain-spec.npz',allow_pickle=False).items()}
        graph=np.load(self.root/'graph.npz',allow_pickle=False)
        values=graph['values'].copy()
        if digest(values)!=self.info['initialWeightHash']:
            raise RuntimeError('Graph fingerprint mismatch')
        self.n=len(graph['body_ids'])
        if all_synapses:
            # Every reconstructed CSR entry has its own independent log gain.
            # This does not add routes, change neurotransmitter signs or claim
            # that biological dopamine reaches all of these synapses.
            edges=np.arange(len(values),dtype=np.int64)
            post=np.repeat(np.arange(self.n,dtype=np.int64),np.diff(graph['crow']))
            self.spec.update(plastic_edges=edges,pre=graph['col'],post=post,
                plastic_rows=np.arange(self.n,dtype=np.int64),post_rank=post,base=values)
            self.info.update(schema=ALL_SYNAPSES_SCHEMA,plasticSynapses=len(edges),
                plasticCells=self.n,plasticMaskHash=digest(edges))
        self.wiring=torch.sparse_csr_tensor(torch.tensor(graph['crow'],device='cuda'),
            torch.tensor(graph['col'],device='cuda'),torch.tensor(values,device='cuda'),
            size=(self.n,self.n),check_invariants=True)
        self.t={k:torch.as_tensor(v,device='cuda') for k,v in self.spec.items()}
        self.state=torch.zeros((self.n,batch),device='cuda')
        self.eligibility=torch.zeros((len(self.t['pre']),batch),device='cuda')
        self.proposal=torch.zeros(len(self.t['pre']),device='cuda')
        self.evoked=torch.zeros((2,batch),device='cuda')
        self.generator=torch.Generator(device='cuda').manual_seed(seed)
        self.noise=torch.zeros((len(self.t['plastic_rows']),batch),device='cuda')
        self.sigma=torch.full((len(self.t['plastic_rows']),1),.012,device='cuda')
        motor_indices=torch.cat([self.t['motor_'+k] for k in ('forward','left','right','interact')])
        self.sigma[torch.searchsorted(self.t['plastic_rows'],motor_indices)]=.065
        self.ticks=0; self.local_samples=0; self.last_modulator=np.zeros(batch)
        self.gains=np.zeros(len(self.t['base']),np.float32)

    @torch.inference_mode()
    def load_gains(self,gains):
        gains=np.asarray(gains,dtype=np.float32)
        if gains.shape!=self.gains.shape or not np.isfinite(gains).all() or np.abs(gains).max()>2.001:
            raise ValueError('Invalid shared synaptic gains')
        self.gains=gains.copy()
        self.wiring.values()[self.t['plastic_edges']]=self.t['base']*torch.exp(torch.as_tensor(gains,device='cuda'))

    @torch.inference_mode()
    def reset(self,indices=None):
        if indices is None: indices=list(range(self.batch))
        self.state[:,indices]=0; self.eligibility[:,indices]=0; self.evoked[:,indices]=0; self.noise[:,indices]=0

    @torch.inference_mode()
    def act(self,observations,explore=True):
        obs=torch.as_tensor(np.asarray(observations).T,device='cuda',dtype=torch.float32)
        drive=torch.zeros_like(self.state)
        drive[self.t['sensory_indices']]=.5*obs[self.t['sensory_channels']]
        if explore and self.ticks%5==0:
            self.noise=torch.randn(self.noise.shape,device='cuda',generator=self.generator)*self.sigma
        elif not explore:
            self.noise.zero_()
        pre=self.state[self.t['pre']].clone() if explore else None
        fluct=torch.zeros_like(self.noise)
        for _ in range(4):
            current=1.5*torch.sparse.mm(self.wiring,self.state*self.t['fast_mask'])+drive+self.t['tonic']
            clean=torch.tanh(torch.relu(current[self.t['plastic_rows']]))
            current[self.t['plastic_rows']]+=self.noise
            target=torch.tanh(torch.relu(current))
            fluct+=(target[self.t['plastic_rows']]-clean)*.25
            self.state.mul_(.75).add_(target,alpha=.25)
        if explore:
            # Local presynaptic activity x postsynaptic exploratory fluctuation.
            # A 1-second eligibility trace bridges the delayed reward.
            pair=(pre.clamp(0,.5)/.1)*(fluct[self.t['post_rank']]/.03)
            self.eligibility.mul_(.95).add_(pair,alpha=.05).clamp_(-3,3)
        rates=torch.stack([self.state[self.t['motor_'+k]].mean(0) for k in ('forward','left','right','interact')]).cpu().numpy()
        self.ticks+=1
        return [{'speed':float(np.clip(80*rates[0,i],0,2)),
                 'turn':float(np.clip(160*(rates[2,i]-rates[1,i]),-2,2)),
                 'interact':bool(rates[3,i]>.025),
                 'rates':dict(zip(('forward','left','right','interact'),map(float,rates[:,i])))} for i in range(self.batch)]

    @torch.inference_mode()
    def reinforce(self,rewards,enabled=True):
        if not enabled:
            self.last_modulator=np.zeros(self.batch); return
        r=torch.as_tensor(rewards,device='cuda',dtype=torch.float32).clamp(-1,1)
        # Synthetic signed reinforcement is delivered to identified DAN populations.
        # The measured evoked rate increment, not reward renamed as dopamine, gates
        # the eligibility trace. Extending this signal to our mask is artificial.
        increments=[]
        for name,pulse in [('appetitive',torch.relu(r)),('aversive',torch.relu(-r))]:
            indices=self.t['dan_'+name]
            before=self.state[indices].clone()
            after=before+.5*pulse[None,:]*(1-before)
            self.state[indices]=after.clamp(0,1)
            increments.append((self.state[indices]-before).mean(0))
        self.evoked.mul_(.8).add_(torch.stack(increments))
        modulator=(self.evoked[0]-self.evoked[1]).clamp(-1,1)
        self.proposal+=(self.eligibility*modulator[None,:]).mean(1)*.12
        self.local_samples+=self.batch
        self.last_modulator=modulator.cpu().numpy()

    @torch.inference_mode()
    def take_proposal(self):
        result=self.proposal.cpu().numpy().copy()
        self.proposal.zero_(); self.local_samples=0
        return result

    def audit(self):
        actual=self.wiring.values().cpu().numpy()
        base=np.load(self.root/'graph.npz')['values']
        mask=np.ones(len(actual),bool); mask[self.spec['plastic_edges']]=False
        expected=self.spec['base']*np.exp(self.gains)
        return {'schema':self.info['schema'],'initialWeightHash':digest(base),'currentWeightHash':digest(actual),
                'eligibleSynapses':len(self.gains),'totalSynapses':len(actual),
                'changedSynapses':int(np.count_nonzero(actual!=base)),
                'frozenSynapsesUnchanged':bool(np.array_equal(actual[mask],base[mask])),
                'sharedWeightsMatch':bool(np.allclose(actual[self.spec['plastic_edges']],expected,rtol=1e-5,atol=1e-9)),
                'finite':bool(np.isfinite(actual).all()),'signsPreserved':bool(np.array_equal(np.sign(actual),np.sign(base))),
                'optimizerPresent':False,'backpropagation':False,'decoderTrained':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--prepare',type=Path,required=True)
    prepare(p.parse_args().prepare)

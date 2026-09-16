"""Brain-only experiment: train existing neuron excitability and synapses.

No new neurons/edges, encoder network, action override, or learned decoder.
The selected engineered sensory projection remains fixed. This is supervised
backpropagation, not a claim about biological plasticity.
"""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from .layout_brain import LayoutConnectome
from .layout_training import focus_batch,sample_batch,evaluate,demonstration_sequences,LABEL_VERSION
from .supervised_joint import raw_readout
from .plastic_brain import digest

class SurrogateReLU(torch.autograd.Function):
    """Exact nonnegative forward; smooth training derivative for silent cells."""
    @staticmethod
    def forward(ctx,current):
        ctx.save_for_backward(current)
        return torch.relu(current)

    @staticmethod
    def backward(ctx,adjoint):
        current,=ctx.saved_tensors
        return adjoint*torch.sigmoid(current/.005)

class ExcitableConnectome(LayoutConnectome):
    def __init__(self,root,initial,tonic=None,odor=False,surrogate=False,gated=False,annotated=False):
        super().__init__(root,initial,stock=True)
        original=self.tonic.detach().clone()
        del self._buffers['tonic']
        self.tonic=torch.nn.Parameter(original if tonic is None else torch.as_tensor(tonic,device=original.device))
        if self.tonic.shape!=original.shape or not torch.isfinite(self.tonic).all():raise ValueError('Invalid excitability')
        self.odor_context=odor
        self.surrogate_training=surrogate
        self.gated=gated
        self.annotated=annotated
        self.interface='local-color-cargo-v3-odor-excitability' if odor else 'local-color-cargo-v3-excitability'
        if gated:
            self.interface+='-gated'
            hidden=torch.ones((self.n,1),device=self.tonic.device,dtype=torch.bool)
            hidden[self.sensory_indices]=False
            for name in ('forward','left','right','interact'):hidden[getattr(self,'motor_'+name)]=False
            self.register_buffer('hidden_neurons',hidden)
        if odor:
            indices=torch.full_like(self.original_sensory_channels,126)
            mask=torch.zeros_like(indices,dtype=torch.float32)
            for side in (24,25):
                cells=torch.where(self.original_sensory_channels==side)[0]
                for cargo in range(3):indices[cells[cargo::3]]=126+cargo
                mask[cells]=1
            self.register_buffer('cargo_odor_indices',indices)
            self.register_buffer('cargo_odor_mask',mask)
        if annotated:
            if odor or gated:raise ValueError('Annotated projection uses its own odor mapping and original neuron response')
            with np.load(Path(root)/'annotated-inputs.npz',allow_pickle=False) as mapping:
                if not np.array_equal(mapping['indices'],self.sensory_indices.cpu().numpy()):raise ValueError('Sensory cell mismatch')
                channels=mapping['channels'].copy()
            if len(channels)!=len(self.sensory_indices) or channels.min()<0 or channels.max()>=129:raise ValueError('Invalid sensory map')
            self.register_buffer('annotated_channels',torch.as_tensor(channels,device=self.tonic.device))
            self.input_channels=129;self.stock_sensing=False;self.interface='annotated-color-cargo-v1'
        self.fixed_hash=self.fingerprint()

    def sensory_drive(self,observations):
        if self.annotated:return .5*observations.T[self.annotated_channels]
        drive=super().sensory_drive(observations)
        if self.odor_context:
            # Identity of carried material stimulates fixed sensory cells. All
            # destinations remain visible simultaneously; no target is selected.
            drive=drive+.5*observations.T[self.cargo_odor_indices]*self.cargo_odor_mask[:,None]
        return drive

    def activation(self,current):
        if self.gated:
            # Alternative rate-neuron response, not claimed biologically fitted.
            # Existing sensory cells and motor readout cells keep their response;
            # the fixed wiring still carries every intermediate computation.
            return torch.where(self.hidden_neurons,.03*torch.sigmoid((current-.01)/.005),super().activation(current))
        if self.surrogate_training and self.training and torch.is_grad_enabled():
            return torch.tanh(SurrogateReLU.apply(current))
        return super().activation(current)

    def checkpoint_hash(self):
        h=hashlib.sha256()
        for parameter in (self.log_gains,self.tonic):h.update(parameter.detach().cpu().numpy().tobytes())
        return h.hexdigest()

@torch.no_grad()
def diagnostic(model,observations,targets):
    predictions=[];weights=model.weights()
    for offset in range(0,len(observations),16):
        _,state=model(torch.as_tensor(observations[offset:offset+16],device='cuda'),128,weights=weights)
        predictions.extend(raw_readout(model,state).cpu().tolist())
    p=np.asarray(predictions);delta=p.reshape(-1,4,3)[:,1:,1]
    target=targets.reshape(-1,4,3)[:,1:,1]
    mask=np.abs(target)>.2
    blind=(targets[:,0]>.79)&(np.abs(targets[:,1])<1e-7)
    centered=(delta-delta.mean(1,keepdims=True)).ravel()
    centered_target=(target-target.mean(1,keepdims=True)).ravel()
    correlation=(float(np.corrcoef(centered,centered_target)[0,1])
                 if np.std(centered)>1e-12 and np.std(centered_target)>1e-12 else None)
    return {'mse':((p-targets)**2).mean(0).tolist(),
            'meanOutput':p.mean(0).tolist(),'meanTarget':targets.mean(0).tolist(),
            'outputRange':[p.min(0).tolist(),p.max(0).tolist()],
            'blindCases':int(blind.sum()),
            'blindMeanSpeed':float(p[blind,0].mean()) if blind.any() else None,
            'blindMeanAbsTurn':float(np.abs(p[blind,1]).mean()) if blind.any() else None,
            'blindExplorationFraction':float(np.mean((p[blind,0]>.5)&(np.abs(p[blind,1])<.15))) if blind.any() else None,
            'directionAccuracy':float(np.mean(np.sign(delta[mask])==np.sign(target[mask]))),
            'cargoTurnRange':float(np.ptp(delta,axis=1).mean()),
            'targetCargoTurnRange':float(np.ptp(target,axis=1).mean()),
            'conditionalTurnCorrelation':correlation}

def main(args):
    if args.trajectory_replay and args.no_replay:raise ValueError('Trajectory replay cannot be disabled')
    if args.replay_heavy and args.no_replay:raise ValueError('Replay cannot be both emphasized and disabled')
    if args.partial_scenes<0:raise ValueError('Partial-scene count must be nonnegative')
    if args.scenes<1 or args.updates<1 or args.save_every<1:raise ValueError('Scene and update counts must be positive')
    if args.cases<1 or args.eval_seconds<1:raise ValueError('Evaluation bounds must be positive')
    if not (args.lr>0 and args.tonic_lr>0):raise ValueError('Learning rates must be positive')
    if args.out.exists():raise FileExistsError('Preserve previous experiments')
    args.out.mkdir(parents=True);torch.set_num_threads(4)
    with np.load(args.candidate,allow_pickle=False) as archive:
        initial=archive['gains'].copy();tonic=archive['tonic'].copy() if 'tonic' in archive.files else None
        interface=str(archive['interface']) if 'interface' in archive.files else None
    model=ExcitableConnectome(args.root,initial,tonic,odor=args.odor,surrogate=args.surrogate,gated=args.gated,annotated=args.annotated)
    if interface is not None and interface!=model.interface:raise ValueError('Checkpoint sensory interface mismatch')
    start_tonic=model.tonic.detach().clone();before=model.checkpoint_hash()
    manifest={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    manifest.update(parentParameterHash=before,sensoryInterface=model.interface,
                    fixedGraphAndInterfaceHash=model.fixed_hash,device=torch.cuda.get_device_name(),
                    trainingDataSeed=7000000,developmentDiagnosticSeed=6800000,
                    labelVersion=LABEL_VERSION,
                    training='supervised backpropagation; not dopamine',externalDecisionNetwork=False)
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    if args.evaluate:
        model.eval().requires_grad_(False)
        report=evaluate(model,args.eval_seed,args.cases,args.eval_seconds,kinds=args.kinds.split(','))
        report['synapticGainsHash']=report['checkpointHash'];report['checkpointHash']=before
        report['excitabilityHash']=digest(model.tonic.detach().cpu().numpy())
        assert before==model.checkpoint_hash()
        (args.out/'evaluation.json').write_text(json.dumps(report,indent=2))
        print('RESULT '+json.dumps(report['summary']),flush=True);return
    rng=np.random.default_rng(7000000)
    x,y=focus_batch(rng,args.scenes,stock=model.stock_sensing);vx,vy=focus_batch(np.random.default_rng(6800000),32,stock=model.stock_sensing)
    if args.partial_scenes:
        parts=[focus_batch(rng,args.partial_scenes,stock=model.stock_sensing,missing=k,varied=True) for k in (0,1,2)]
        x=np.concatenate([x,*[part[0] for part in parts]])
        y=np.concatenate([y,*[part[1] for part in parts]])
        partial_checks=[focus_batch(np.random.default_rng(7300000+k),32,stock=model.stock_sensing,missing=k,varied=True) for k in (0,1,2)]
        px=np.concatenate([part[0] for part in partial_checks]);py=np.concatenate([part[1] for part in partial_checks])
    # Replay real near-box interaction, blind exploration and variable layouts.
    rx,ry=sample_batch(rng,1024,stock=model.stock_sensing,contrast=True)
    x=torch.as_tensor(x,device='cuda');y=torch.as_tensor(y,device='cuda')
    rx=torch.as_tensor(rx,device='cuda');ry=torch.as_tensor(ry,device='cuda')
    if args.trajectory_replay:
        tx,ty,demonstrations=demonstration_sequences(rng,stock=model.stock_sensing)
        tx=torch.as_tensor(tx,device='cuda');ty=torch.as_tensor(ty,device='cuda')
        (args.out/'demonstrations.json').write_text(json.dumps(demonstrations,indent=2))
        print('OFFLINE_TEACHER_DATA '+json.dumps(demonstrations),flush=True)
    optimizer=torch.optim.Adam([{'params':[model.log_gains],'lr':args.lr},
                                 {'params':[model.tonic],'lr':args.tonic_lr}],eps=1e-10)
    history=[];began=time.perf_counter()
    print('INITIAL '+json.dumps(diagnostic(model,vx,vy)),flush=True)
    if args.partial_scenes:print('INITIAL_GENERAL_LAYOUT '+json.dumps(diagnostic(model,px,py)),flush=True)
    for update in range(1,args.updates+1):
        replay=not args.no_replay and (update%4!=0 if args.replay_heavy else update%4==0)
        xx,yy=(rx,ry) if replay else (x,y)
        scene_count=len(xx)//4
        scenes=np.arange(4)%scene_count if scene_count<=4 and not replay else rng.integers(0,scene_count,4)
        indices=torch.as_tensor((4*scenes[:,None]+np.arange(4)).ravel(),device='cuda')
        optimizer.zero_grad(set_to_none=True);weights=model.weights()
        if replay and args.trajectory_replay:
            # Separate demonstration windows. Burn-in uses changing observations,
            # not a static photograph; the final eight frames carry gradients.
            starts=torch.as_tensor(rng.integers(0,len(tx)-24+1,16),device='cuda')
            columns=torch.as_tensor(rng.integers(0,tx.shape[1],16),device='cuda')
            state=None
            with torch.no_grad():
                for frame in range(16):_,state=model(tx[starts+frame,columns],4,state,weights)
            errors=[]
            for frame in range(16,24):
                _,state=model(tx[starts+frame,columns],4,state,weights)
                errors.append((raw_readout(model,state)-ty[starts+frame,columns]).square().mean(0))
            loss=torch.stack(errors).mean(0).sum()
        else:
            with torch.no_grad():_,state=model(xx[indices],64,weights=weights)
            _,state=model(xx[indices],32,state,weights)
            pred=raw_readout(model,state);error=pred-yy[indices]
            delta=error.reshape(-1,4,3)[:,1:,:2]
            contrast=(delta-delta.mean(1,keepdim=True)).square().mean()
            loss=error.square().mean(0).sum()+(0 if replay else 2*contrast)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward()
        if any(not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Nonfinite gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        with torch.no_grad():model.log_gains.clamp_(-2,2);model.tonic.clamp_(-.1,.1)
        if update==1 or update%20 in (0,1):
            row={'update':update,'task':('demonstration sequence' if args.trajectory_replay else 'replay') if replay else 'cargo contrast','loss':float(loss.detach()),'seconds':time.perf_counter()-began}
            history.append(row);print(json.dumps(row),flush=True)
        if update%args.save_every==0 or update==args.updates:
            np.savez_compressed(args.out/f'candidate-{update}.npz',gains=model.log_gains.detach().cpu().numpy(),
                                tonic=model.tonic.detach().cpu().numpy(),interface=np.asarray(model.interface))
            check=diagnostic(model,vx,vy);history.append({'update':update,'diagnostic':check})
            print('FOCUS_CHECK '+json.dumps({'update':update,**check}),flush=True)
            if args.partial_scenes:
                partial=diagnostic(model,px,py);history.append({'update':update,'generalLayout':partial})
                print('GENERAL_LAYOUT_CHECK '+json.dumps({'update':update,**partial}),flush=True)
            if len(x)//4<=64:
                fitted=diagnostic(model,x.cpu().numpy(),y.cpu().numpy())
                history.append({'update':update,'trainingScenes':fitted})
                print('TRAIN_SCENES_CHECK '+json.dumps({'update':update,**fitted}),flush=True)
            (args.out/'history.json').write_text(json.dumps(history,indent=2))
    audit=model.audit(initial)
    audit['fixedGraphSensoryAndMotorDecoderUnchanged']=audit.pop('fixedGraphSensoryDecoderDynamicsUnchanged')
    result={'audit':audit,'neuronalExcitabilityTrained':True,
            'changedNeurons':int(torch.count_nonzero(model.tonic.detach()!=start_tonic)),
            'checkpointHash':model.checkpoint_hash(),'sensoryInterface':model.interface,
            'decoderTrained':False,'newNeuronsOrEdges':False,'requiresFrozenVerification':True,
            'surrogateGradientTraining':args.surrogate,
            'alternativeHiddenNeuronResponse':args.gated,
            'curriculumScenes':args.scenes,'staticTeacherReplay':not args.no_replay and not args.trajectory_replay,
            'partialScenesPerMissingCount':args.partial_scenes,
            'offlineDemonstrationReplay':args.trajectory_replay,
            'replayUpdateFraction':0 if args.no_replay else (.75 if args.replay_heavy else .25),
            'updates':args.updates,'seconds':time.perf_counter()-began}
    (args.out/'result.json').write_text(json.dumps(result,indent=2));print('TRAINING_FINISHED',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','candidate','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--lr',type=float,default=.01);p.add_argument('--tonic-lr',type=float,default=.0001)
    p.add_argument('--odor',action='store_true')
    p.add_argument('--surrogate',action='store_true')
    p.add_argument('--gated',action='store_true')
    p.add_argument('--annotated',action='store_true')
    p.add_argument('--scenes',type=int,default=1024);p.add_argument('--no-replay',action='store_true')
    p.add_argument('--partial-scenes',type=int,default=0)
    p.add_argument('--trajectory-replay',action='store_true')
    p.add_argument('--replay-heavy',action='store_true')
    p.add_argument('--updates',type=int,default=240);p.add_argument('--save-every',type=int,default=80)
    p.add_argument('--evaluate',action='store_true');p.add_argument('--eval-seconds',type=int,default=240)
    p.add_argument('--cases',type=int,default=4);p.add_argument('--eval-seed',type=int,default=6300000)
    p.add_argument('--kinds',default='rotated,permuted,compact,wide')
    main(p.parse_args())

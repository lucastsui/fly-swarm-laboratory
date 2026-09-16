"""Bounded grip discrimination on actual operations, anchored parent motion.

One canonical Spark1 optimizer; preserve C60 Adam/RNG, explicitly restore10x
rates for40updates. Only original signed synaptic gains and excitabilities
change. Exact ReLU;96 differentiated frames,32 loss frames. No physics/sensory/
decoder modification, teacher action, additional network or dopamine claim.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .layout_correction_curriculum_train import source_run, restore_state
from .layout_operation_sampling import OperationCache
from .layout_recovery_brain import load_model, save_model
from .layout_recovery_train import recurrent_predictions, optimize
from .layout_grip_objective import threshold_grip_loss
from .layout_demonstration_train import training_scores
from .layout_demonstration_step_probe import frozen_predictions
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json


def operation_heads(predictions, labels, parent_motion):
    if predictions.shape != labels.shape or parent_motion.shape != predictions.shape[:-1]+(2,):
        raise ValueError('Aligned original labels and fixed parent motion required')
    motion = (predictions[...,:2]-parent_motion).square().mean((0,1))
    # Equal actor-context weight. A long positive episode cannot overwhelm a
    # loaded-return counterexample, nor can grip-off dominate rare pickups.
    handling = torch.stack([threshold_grip_loss(predictions[:,i,2], labels[:,i,2], .1)
                            for i in range(predictions.shape[1])]).mean()
    return torch.stack((motion[0],motion[1],handling))


@torch.no_grad()
def fit(model, bank):
    before = model.checkpoint_hash(); saved_mode = model.training
    ids = bank.diagnostic_indexes(); predictions, targets, anchors = [], [], []
    try:
        model.eval()
        for start in range(0,len(ids),16):
            x,y,state,motion = bank.batch(ids[start:start+16],model.tonic.device)
            predictions.append(frozen_predictions(model,x,state,64))
            targets.append(y[64:]); anchors.append(motion)
        p,y,a = (torch.cat(v,dim=1) for v in (predictions,targets,anchors))
        scores = training_scores(p,y)
        return {'parameterHash':before,'windowIndexes':ids,'trainingSubsetOnly':True,
                'parentMotionMSE':(p[...,:2]-a).square().mean((0,1)).cpu().tolist(),
                'objectiveHeads':operation_heads(p,y,a).cpu().tolist(),**scores}
    finally:
        model.train(saved_mode)
        if model.checkpoint_hash()!=before:
            raise AssertionError('Fit diagnostic changed brain')


def train(model, bank, optimizer, rng, updates, record):
    if type(updates) is not int or not 1<=updates<=40:
        raise ValueError('Hard40-update prefix age bound required')
    began=time.perf_counter(); history=[]
    model.train().requires_grad_(True)
    for update in range(61,61+updates):
        optimizer.zero_grad(set_to_none=True)
        x,y,state,motion,selection=bank.sample(rng,model.tonic.device)
        if x.shape!=(96,16,297) or y.shape!=(96,16,3) or motion.shape!=(32,16,2):
            raise ValueError('Exactly16 independent full96-frame actors required')
        p=recurrent_predictions(model,x,state,64,model.weights(),gradient_start=0)
        heads=operation_heads(p,y[64:],motion); norms={}
        loss=optimize(model,optimizer,heads,(8.,4.,8.),True,norms)
        row={'update':update,'lossBeforeUpdate':loss,'objectiveHeadsBeforeUpdate':heads.detach().cpu().tolist(),
             'gradientNormsBeforeClipping':norms,'seconds':time.perf_counter()-began,
             'sampling':selection,'isServiceEvidence':False}
        del x,y,state,motion,p,heads
        if update%20==0 or update==60+updates:
            row['trainingFit']=fit(model,bank); row['parameterHash']=model.checkpoint_hash()
            record(update,row)
        history.append(row)
        print('OPERATION_GRIP_UPDATE '+json.dumps(row),flush=True)
    return history


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve canonical runs')
    if type(args.updates) is not int or not 1<=args.updates<=40:
        raise ValueError('Bounded correction experiment required')
    source,parent,_=source_run(args.source)
    torch.set_num_threads(4)
    candidate=source/'candidate-60.npz'
    model=load_model(args.root,candidate).eval()
    before=model.checkpoint_hash()
    if before!=parent['parentParameterHash'] or model.fixed_hash!=parent['fixedHash']:
        raise ValueError('Wrong retained C60')
    bank=OperationCache(args.cache,args.dataset,before,model.fixed_hash,model.n)
    if (bank.manifest['candidateFileHash']!=file_hash(candidate)
            or bank.manifest['prefixCanonicalRun']!='recovery-v22-service-curriculum'
            or bank.manifest['prefixCanonicalVersion']!=60):
        raise ValueError('Wrong original versioned operation prefix')
    optimizer,rng=restore_state(model,torch.load(source/'optimizer-60.pt',map_location='cpu',weights_only=True),
                                parent['learningRates'])
    previous_rates=[g['lr'] for g in optimizer.param_groups]
    for group in optimizer.param_groups: group['lr']*=10
    paths=[source/n for n in ('manifest.json','result.json','status.json','fit-60.json','candidate-60.npz','optimizer-60.pt')]
    paths += [args.cache/'manifest.json',args.cache/'cache.npz',args.dataset/'manifest.json']
    paths += [args.dataset/e['file'] for e in bank.manifest['datasetFiles']]
    inputs={str(p.resolve()):file_hash(p) for p in paths}
    names=set(parent['sourceHashes'])|set(bank.manifest['sourceHashes'])|{
        'layout_operation_grip_train.py','layout_operation_sampling.py','layout_correction_curriculum_train.py',
        'layout_demonstration_step_probe.py'}
    sources={n:file_hash(Path(__file__).parent/n) for n in sorted(names)}
    initial=fit(model,bank)
    if max(initial['parentMotionMSE'])>1e-8:
        raise ValueError('Parent anchor predictions do not match exact frozen parent')
    args.out.mkdir(parents=True)
    manifest={'experiment':'physical-operation-grip-parent-motion-v1','startUpdate':60,'finalUpdate':60+args.updates,
        'sourceRun':str(source),'parentParameterHash':before,'fixedHash':model.fixed_hash,'interface':model.interface,
        'sourceHashes':sources,'inputFileHashes':inputs,'optimizerMomentsPreserved':True,'samplerRNGPreserved':True,
        'previousLearningRates':previous_rates,'explicitRateFactor':10,'learningRates':[g['lr'] for g in optimizer.param_groups],
        'canonicalOptimizer':'Spark1 only','gradientFrames':96,'lossFrames':32,'maximumPrefixAgeUpdates':40,
        'objective':'8*parent-speed-MSE+4*parent-turn-MSE+8*per-actor balanced threshold grip hinge(margin0.1)',
        'correctionContexts':list(bank.groups),'allContextsEveryUpdate':True,
        'labelsUnchanged':True,'parentMotionTargetsAreNotTeacherLocomotion':True,
        'teacherAtInference':False,'externalDecisionNetwork':False,'decoderTrained':False,'dopamineLearning':False,
        'learning':'supervised recurrent backpropagation; exact ReLU derivative','isServiceEvidence':False}
    atomic_json(args.out/'manifest.json',manifest)
    initial_gains=model.log_gains.detach().cpu().numpy().copy(); initial_tonic=model.tonic.detach().cpu().numpy().copy()
    last_saved=60
    def record(update,row):
        nonlocal last_saved
        save_model(args.out/f'candidate-{update}.npz',model)
        torch.save({'optimizer':optimizer.state_dict(),'updates':update,'rng':rng.bit_generator.state},args.out/f'optimizer-{update}.pt')
        atomic_json(args.out/f'fit-{update}.json',row)
        atomic_json(args.out/'status.json',{'finished':False,**row}); last_saved=update
    try:
        record(60,{'update':60,'parameterHash':before,'trainingFit':initial,'isServiceEvidence':False})
        history=train(model,bank,optimizer,rng,args.updates,record)
        if (any(file_hash(Path(p))!=h for p,h in inputs.items())
                or any(file_hash(Path(__file__).parent/n)!=h for n,h in sources.items())):
            raise ValueError('Source/data changed during training')
        result={'startUpdate':60,'updates':60+args.updates,'history':history,'finalParameterHash':model.checkpoint_hash(),
                'audit':model.audit(initial_gains),'changedNeurons':int(np.count_nonzero(model.tonic.detach().cpu().numpy()!=initial_tonic)),
                'sourceFilesUnchanged':True,'isServiceEvidence':False}
        atomic_json(args.out/'result.json',result)
        atomic_json(args.out/'status.json',{'finished':True,**history[-1]})
        print('OPERATION_GRIP_FINISHED '+json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)
    except BaseException as error:
        atomic_json(args.out/'failure.json',{'type':type(error).__name__,'error':str(error),'lastSaved':last_saved})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','source','cache','dataset','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--updates',type=int,default=40)
    main(p.parse_args())

"""One bounded canonical optimizer: full-service demos plus actual mistakes.

Branch from the completed V22 run's retained C60 Adam/RNG unchanged after its
C140 regressed on frozen wide layouts. Both kinds of recurrent prefix
are refreshed at that exact parent. Three demonstration updates alternate
with one learner-only correction update; their label versions stay explicit.
No teacher, input selector, new network or decoder is introduced at inference.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .layout_recovery_brain import load_model,save_model
from .layout_cooldown_continuation import restore_adam
from .layout_demonstration_train import DemonstrationCache,demonstration_fit
from .layout_service_balanced_sampling import ServiceBalancedDemonstrations
from .layout_service_curriculum_train import train_curriculum
from .layout_correction_sampling import CorrectionCache,MixedCorrectionCurriculum
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json


def source_run(folder):
    p=Path(folder).resolve()
    m,r,s=[json.loads((p/n).read_text()) for n in ('manifest.json','result.json','status.json')]
    if (m.get('experiment')!='service-balanced-cooldown-curriculum-v1'
            or m['startUpdate']!=60 or m['finalUpdate']!=140 or r['updates']!=140
            or r['startUpdate']!=60 or s.get('finished') is not True or s['update']!=140
            or r['finalParameterHash']!=s['parameterHash']
            or [x['update'] for x in r['history']]!=list(range(61,141))
            or not all(np.isfinite(x['lossBeforeUpdate']) for x in r['history'])
            or r.get('sourceFilesUnchanged') is not True
            or not all(r['audit'][k] for k in ('finite','signsPreserved','fixedGraphSensoryDecoderDynamicsUnchanged'))):
        raise ValueError('Require completed original V22 with unchanged canonical optimizer')
    if ((p/'failure.json').exists() or any(file_hash(Path(__file__).parent/n)!=h for n,h in m['sourceHashes'].items())
            or any(file_hash(Path(n))!=h for n,h in m['inputFileHashes'].items())):
        raise ValueError('Original V22 inputs/source changed')
    fit=json.loads((p/'fit-60.json').read_text())
    if fit['parameterHash']!=m['parentParameterHash']:
        raise ValueError('Parent publication mismatch')
    return p,m,r


def restore_state(model,payload,rates):
    if set(payload)!={'optimizer','updates','rng'} or payload['updates']!=60:
        raise ValueError('Exact V22 Adam/RNG checkpoint required')
    opt=torch.optim.Adam([{'params':[model.log_gains],'lr':rates[0],'eps':1e-14},
                          {'params':[model.tonic],'lr':rates[1],'eps':1e-10}])
    restore_adam(opt,{'optimizer':payload['optimizer'],'updates':payload['updates']},60)
    rng=np.random.default_rng()
    rng.bit_generator.state=payload['rng']
    return opt,rng


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve all canonical runs')
    if type(args.updates) is not int or not 1<=args.updates<=40:
        raise ValueError('Bounded1..40 correction experiment required')
    source,parent,result=source_run(args.source)
    torch.set_num_threads(4)
    model=load_model(args.root,source/'candidate-60.npz').eval().requires_grad_(False)
    before=model.checkpoint_hash()
    if before!=parent['parentParameterHash'] or model.fixed_hash!=parent['fixedHash']:
        raise ValueError('Wrong V22 parent brain')
    bank=DemonstrationCache(args.cache,before,model.fixed_hash,64,32,model.n)
    if (bank.manifest.get('generatorVersion')!='independent-batched-full-prefix-v1'
            or bank.manifest['candidateFileHash']!=file_hash(source/'candidate-60.npz')
            or bank.manifest['sourceHash']!=file_hash(Path(__file__).parent/'layout_batched_prefix_cache.py')
            or any(file_hash(Path(__file__).parent/n)!=h for n,h in bank.manifest['helperSourceHashes'].items())):
        raise ValueError('Demonstration prefix not freshly encoded with exact parent')
    demo=ServiceBalancedDemonstrations(bank,args.dataset,args.labels)
    correction=CorrectionCache(args.correction_cache,args.correction_dataset,before,model.fixed_hash,model.n)
    cm=correction.manifest
    if (cm['prefixCanonicalRun']!='recovery-v22-service-curriculum' or cm['prefixCanonicalVersion']!=60
            or cm['candidateFileHash']!=file_hash(source/'candidate-60.npz')):
        raise ValueError('Correction prefix canonical parent mismatch')
    sampler=MixedCorrectionCurriculum(demo,correction)
    optimizer,rng=restore_state(model,torch.load(source/'optimizer-60.pt',map_location='cpu',weights_only=True),
                                parent['learningRates'])
    files=[source/n for n in ('manifest.json','result.json','status.json','fit-60.json',
                              'candidate-60.npz','optimizer-60.pt')]
    for folder in (args.cache,args.correction_cache):
        files.extend(folder/n for n in ('manifest.json','cache.npz'))
    for folder in (args.dataset,args.correction_dataset,args.labels):
        files.append(folder/'manifest.json')
        files.extend(folder/e['file'] for e in json.loads((folder/'manifest.json').read_text())['episodes'])
    inputs={str(p.resolve()):file_hash(p) for p in files}
    names=set(parent['sourceHashes'])|set(cm['sourceHashes'])|set(bank.manifest['helperSourceHashes'])
    names|={'layout_correction_curriculum_train.py','layout_correction_sampling.py'}
    sources={n:file_hash(Path(__file__).parent/n) for n in sorted(names)}
    args.out.mkdir(parents=True)
    m={'experiment':'service-plus-learner-corrections-v1','sourceRun':str(source),'startUpdate':60,
       'selectedEarlierCheckpointReason':'V22 C140 wide940 regressed to0/8 sustained; retain experimental C60 branch',
       'finalUpdate':60+args.updates,'parentParameterHash':before,'fixedHash':model.fixed_hash,
       'interface':model.interface,'optimizerMomentsPreserved':True,'samplerRNGPreserved':True,
       'learningRates':[g['lr'] for g in optimizer.param_groups],'sourceHashes':sources,'inputFileHashes':inputs,
       'canonicalOptimizer':'Spark1 only','gradientFrames':96,'lossFrames':32,'maximumPrefixAgeUpdates':40,
       'mix':'3 service-balanced demonstration updates then1 learner-only correction update',
       'demonstrationSampling':demo.record,'correctionSampling':correction.record,
       'differentLabelVersionsExplicit':True,'labelsModified':False,'onlineCorrectionsUsed':False,
       'correctionsUsed':True,'behaviorCheckpointMayDifferFromPrefix':True,
       'teacherAtInference':False,'externalDecisionNetwork':False,'decoderTrained':False,
       'dopamineLearning':False,'learning':'supervised recurrent backpropagation; exact ReLU derivative',
       'objective':'4*teacher-speed+2*teacher-turn+2*balanced grip (versioned by data source)',
       'isServiceEvidence':False}
    atomic_json(args.out/'manifest.json',m)
    initial_gains=model.log_gains.detach().cpu().numpy().copy()
    initial_tonic=model.tonic.detach().cpu().numpy().copy()
    last_saved=60
    def record(update,row):
        nonlocal last_saved
        row['correctionTrainingFit']=demonstration_fit(model,correction,64)
        row['demonstrationUpdates']=sampler.samples-sampler.correction_samples
        row['correctionUpdates']=sampler.correction_samples
        save_model(args.out/f'candidate-{update}.npz',model)
        torch.save({'optimizer':optimizer.state_dict(),'updates':update,'rng':rng.bit_generator.state},
                   args.out/f'optimizer-{update}.pt')
        atomic_json(args.out/f'fit-{update}.json',row)
        atomic_json(args.out/'status.json',{'finished':False,**row})
        last_saved=update
    try:
        record(60,{'update':60,'parameterHash':before,'trainingFit':demonstration_fit(model,demo,64),
                    'isServiceEvidence':False})
        history=train_curriculum(model,sampler,optimizer,rng,60,args.updates,record)
        if (any(file_hash(Path(n))!=h for n,h in inputs.items())
                or any(file_hash(Path(__file__).parent/n)!=h for n,h in sources.items())):
            raise ValueError('Canonical source/data changed during corrections')
        r={'startUpdate':60,'updates':60+args.updates,'history':history,
           'finalParameterHash':model.checkpoint_hash(),'audit':model.audit(initial_gains),
           'changedNeurons':int(np.count_nonzero(model.tonic.detach().cpu().numpy()!=initial_tonic)),
           'demonstrationUpdates':sampler.samples-sampler.correction_samples,
           'correctionUpdates':sampler.correction_samples,'sourceFilesUnchanged':True,'isServiceEvidence':False}
        atomic_json(args.out/'result.json',r)
        atomic_json(args.out/'status.json',{'finished':True,**history[-1]})
        print('CORRECTION_CURRICULUM_FINISHED '+json.dumps({k:v for k,v in r.items() if k!='history'}),flush=True)
    except BaseException as error:
        atomic_json(args.out/'failure.json',{'type':type(error).__name__,'error':str(error),'lastSaved':last_saved})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','source','dataset','labels','cache','correction-dataset','correction-cache','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--updates',type=int,default=40)
    main(p.parse_args())

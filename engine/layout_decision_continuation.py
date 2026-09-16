"""A bounded 100->140 continuation, not a fresh optimizer or a new objective.

Refresh full-history prefixes at the published C100; preserve ORIGINAL C60
motion anchors, original labels/sensory histories, rates and Adam/RNG state.
All decisions remain in the signed connectome with unchanged embodiment.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .layout_decision_resume import source_run, restore_state, AnchoredOperationCache
from .layout_operation_sampling import OperationCache
from .layout_operation_focus import VERSION, focused_heads, focus_fit
from .layout_recovery_brain import load_model, save_model
from .layout_recovery_train import recurrent_predictions, optimize
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json

EXPERIMENT = 'decision-grip-fixed-anchor-c100-continuation-v1'


def train_segment(model, bank, optimizer, rng, updates, record):
    if type(updates) is not int or not 1 <= updates <= 40:
        raise ValueError('Hard40-update refreshed-prefix bound required')
    began = time.perf_counter(); history = []
    model.train().requires_grad_(True)
    for update in range(101,101+updates):
        optimizer.zero_grad(set_to_none=True)
        x,y,state,motion,selection = bank.sample(rng,model.tonic.device)
        if x.shape != (96,16,297) or y.shape != (96,16,3) or motion.shape != (32,16,2):
            raise ValueError('Exactly16 independent full96-frame actors required')
        p = recurrent_predictions(model,x,state,64,model.weights(),gradient_start=0)
        heads = focused_heads(p,y[64:],motion,selection['selections']); norms = {}
        loss = optimize(model,optimizer,heads,(8.,4.,8.),True,norms)
        row = {'update':update,'lossBeforeUpdate':loss,'objectiveHeadsBeforeUpdate':heads.detach().cpu().tolist(),
               'gradientNormsBeforeClipping':norms,'seconds':time.perf_counter()-began,'sampling':selection,
               'gripObjective':VERSION,'originalMotionAnchorHash':bank.manifest['motionAnchorParameterHash'],
               'isServiceEvidence':False}
        del x,y,state,motion,p,heads
        if update%20 == 0 or update == 100+updates:
            row['trainingFit'] = focus_fit(model,bank); row['parameterHash'] = model.checkpoint_hash()
            record(update,row)
        history.append(row)
        print('DECISION_CONTINUATION_UPDATE '+json.dumps(row),flush=True)
    return history


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve canonical runs')
    if type(args.updates) is not int or not 1 <= args.updates <= 40:
        raise ValueError('Bounded continuation required')
    source,parent,result,saved_fit = source_run(args.source)
    # Original data/prefix provenance must match the actual source optimizer run.
    for p in (args.original_cache/'manifest.json',args.original_cache/'cache.npz',args.dataset/'manifest.json'):
        if parent['inputFileHashes'].get(str(p.resolve())) != file_hash(p):
            raise ValueError('Changed original C60 motion anchor/data')
    torch.set_num_threads(4)
    candidate = source/'candidate-100.npz'
    model = load_model(args.root,candidate).eval().requires_grad_(False)
    before = model.checkpoint_hash()
    if before != result['finalParameterHash'] or model.fixed_hash != parent['fixedHash']:
        raise ValueError('Wrong published C100')
    original = OperationCache(args.original_cache,args.dataset,parent['parentParameterHash'],model.fixed_hash,model.n)
    fresh = OperationCache(args.refreshed_cache,args.dataset,before,model.fixed_hash,model.n)
    if (fresh.manifest['candidateFileHash'] != file_hash(candidate)
            or fresh.manifest['prefixCanonicalRun'] != source.name or fresh.manifest['prefixCanonicalVersion'] != 100):
        raise ValueError('Prefix must be refreshed at original published C100')
    original_prefix_fit = focus_fit(model,original)
    for key in ('objectiveHeads','parentMotionMSE','focusRecall','focusFalsePositiveRate'):
        if not np.allclose(original_prefix_fit[key],saved_fit['trainingFit'][key],rtol=1e-5,atol=1e-6):
            raise ValueError('Original C60 anchors do not reconstruct saved C100 fit')
    exact_prefix_fit = focus_fit(model,fresh)
    if max(exact_prefix_fit['parentMotionMSE']) > 1e-8:
        raise ValueError('Refreshed C100 prefix does not reconstruct its own motion targets')
    bank = AnchoredOperationCache(fresh,original)
    initial = focus_fit(model,bank)
    optimizer,rng = restore_state(model,torch.load(source/'optimizer-100.pt',map_location='cpu',weights_only=True),
                                  parent['learningRates'])
    paths = [source/n for n in ('manifest.json','result.json','status.json','fit-100.json','candidate-100.npz','optimizer-100.pt')]
    for folder in (args.original_cache,args.refreshed_cache):
        paths += [folder/'manifest.json',folder/'cache.npz']
    paths += [args.dataset/'manifest.json']+[args.dataset/e['file'] for e in bank.manifest['datasetFiles']]
    inputs = {str(p.resolve()):file_hash(p) for p in paths}
    names = set(parent['sourceHashes'])|set(fresh.manifest['sourceHashes'])|{
        'layout_decision_resume.py','layout_decision_continuation.py','layout_cooldown_continuation.py'}
    sources = {n:file_hash(Path(__file__).parent/n) for n in sorted(names)}
    args.out.mkdir(parents=True)
    manifest = {'experiment':EXPERIMENT,'sourceRun':str(source),'startUpdate':100,'finalUpdate':100+args.updates,
        'parentParameterHash':before,'fixedHash':model.fixed_hash,'interface':model.interface,
        'sourceHashes':sources,'inputFileHashes':inputs,'optimizerMomentsPreserved':True,'samplerRNGPreserved':True,
        'learningRates':[g['lr'] for g in optimizer.param_groups],'learningRatesUnchanged':True,
        'canonicalOptimizer':'Spark1 only','gradientFrames':96,'motionLossFrames':32,'gripLossFrames':1,
        'gripObjective':VERSION,'maximumPrefixAgeUpdates':40,'fullHistoryRefreshed':True,
        'originalMotionAnchorHash':parent['parentParameterHash'],'originalMotionTargetsPreserved':True,
        'sameDatasetWindowsLabelsAndObjective':True,'originalPrefixFit':original_prefix_fit,
        'refreshedCandidateMotionIdentityFit':exact_prefix_fit,'refreshedOriginalAnchorFit':initial,
        'teacherAtInference':False,'externalDecisionNetwork':False,'decoderTrained':False,'dopamineLearning':False,
        'learning':'supervised recurrent backpropagation; exact ReLU derivative','isServiceEvidence':False}
    atomic_json(args.out/'manifest.json',manifest)
    initial_gains = model.log_gains.detach().cpu().numpy().copy()
    initial_tonic = model.tonic.detach().cpu().numpy().copy(); last_saved = 100
    def record(update,row):
        nonlocal last_saved
        save_model(args.out/f'candidate-{update}.npz',model)
        torch.save({'optimizer':optimizer.state_dict(),'updates':update,'rng':rng.bit_generator.state},
                   args.out/f'optimizer-{update}.pt')
        atomic_json(args.out/f'fit-{update}.json',row)
        atomic_json(args.out/'status.json',{'finished':False,**row}); last_saved = update
    try:
        record(100,{'update':100,'parameterHash':before,'trainingFit':initial,'isServiceEvidence':False})
        history = train_segment(model,bank,optimizer,rng,args.updates,record)
        if (any(file_hash(Path(p)) != h for p,h in inputs.items())
                or any(file_hash(Path(__file__).parent/n) != h for n,h in sources.items())):
            raise ValueError('Original source/anchor/data changed during continuation')
        report = {'startUpdate':100,'updates':100+args.updates,'history':history,
                  'finalParameterHash':model.checkpoint_hash(),'audit':model.audit(initial_gains),
                  'changedNeurons':int(np.count_nonzero(model.tonic.detach().cpu().numpy()!=initial_tonic)),
                  'sourceFilesUnchanged':True,'isServiceEvidence':False}
        atomic_json(args.out/'result.json',report)
        atomic_json(args.out/'status.json',{'finished':True,**history[-1]})
        print('DECISION_CONTINUATION_FINISHED '+json.dumps({k:v for k,v in report.items() if k!='history'}),flush=True)
    except BaseException as error:
        atomic_json(args.out/'failure.json',{'type':type(error).__name__,'error':str(error),'lastSaved':last_saved})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root','source','original-cache','refreshed-cache','dataset','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--updates',type=int,default=40)
    main(p.parse_args())

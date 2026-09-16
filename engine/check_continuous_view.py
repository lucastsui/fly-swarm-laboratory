"""Bounded, accelerated integration check; never trains or changes weights."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from .service_viewer import ServiceViewer
from .plastic_brain import digest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=7200)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Preserve previous checks')
    root=Path(__file__).resolve().parents[1]
    viewer=ServiceViewer(SimpleNamespace(root=root/'.runtime/dopamine-haul',
        candidate=root/'.runtime/service-training/sequence-lr0005/candidate-200.npz',
        metadata=root.parent.parent/'outputs/assembly-line-service-checkpoint.json',
        evidence=root.parent/'service-comparison',seed=5500000,flies=1),start_thread=False)
    original_world=viewer.world
    viewer.policy.reset=Mock(side_effect=AssertionError('Unexpected neural reset'))
    milestones=[]
    for step in range(args.steps):
        viewer.tick()
        assert viewer.world is original_world
        if (step+1)%1200==0:
            row={'steps':viewer.ticks,'seconds':viewer.world.steps*.05,
                 'products':viewer.products,'pickups':viewer.pickups,'seed':viewer.seed}
            milestones.append(row);print(json.dumps(row),flush=True)
    unchanged=digest(viewer.model.log_gains.detach().cpu().numpy())==viewer.weight_hash
    assert unchanged and viewer.world.steps==args.steps and viewer.seed==5500000
    viewer.policy.reset.assert_not_called()
    report={'steps':args.steps,'sameWorld':True,'neuralResets':0,'seed':viewer.seed,
            'weightHash':viewer.weight_hash,'weightsUnchanged':unchanged,'learning':False,
            'continuousSupply':True,'products':viewer.products,'milestones':milestones,
            'motionSampleCount':len(viewer.motion_samples)}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2))
    print('CONTINUOUS_VIEW_CHECK_PASSED',flush=True)

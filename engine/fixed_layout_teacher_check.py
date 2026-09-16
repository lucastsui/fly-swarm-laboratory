"""CPU label-policy feasibility on the user's layout, NOT brain performance."""
import argparse
import json
from pathlib import Path
from .fixed_layout_curriculum import load_layout
from .layout_recovery_world import RecoveryWorld
from .layout_recovery_teacher import LocalTeacher
from .layout_recovery_eval import service_metrics
from .layout_recovery_protocol import atomic_json

def main(args):
    if args.out.exists(): raise FileExistsError('Preserve diagnostic')
    positions, digest = load_layout(args.layout_file)
    trials=[]
    for index in range(args.cases):
        random_start = bool(index % 2)
        world=RecoveryWorld(args.seed+index,positions=positions,kind='fixed-current',random_starts=random_start)
        teachers=[LocalTeacher() for _ in range(4)]
        times=[]
        for tick in range(args.seconds*20):
            motors=[]
            for teacher,agent in zip(teachers,world.agents):
                label=teacher.label(agent)
                motors.append({'speed':float(label[0]),'turn':float(label[1]),'interact':bool(label[2]>1.)})
            old=world.deliveries;world.advance(motors)
            times.extend([(tick+1)*.05]*(world.deliveries-old))
        trial={'seed':args.seed+index,'randomStarts':random_start,'products':world.deliveries,
               'returns':world.returns,'deliveryTimes':times,**service_metrics(times,args.seconds)}
        trials.append(trial);print(json.dumps(trial),flush=True)
    atomic_json(args.out,{'teacherActions':True,'brainWasUsed':False,'isBrainEvidence':False,
                          'layoutHash':digest,'seconds':args.seconds,'trials':trials})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--layout-file',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--cases',type=int,default=4)
    p.add_argument('--seconds',type=int,default=600);p.add_argument('--seed',type=int,default=13000000)
    main(p.parse_args())

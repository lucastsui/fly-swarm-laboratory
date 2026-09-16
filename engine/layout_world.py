"""Random-layout factory with local color vision and cargo proprioception.

No destination vector, global map, target selection or action override enters
the sensory interface. Box identity is visible only inside the sensing radius.
"""
import math
import numpy as np
from .swarm_world import SwarmWorld
from .haul_world import wrap

def validate_layout(positions):
    p=np.asarray(positions,dtype=float)
    if p.shape!=(4,2) or not np.isfinite(p).all():raise ValueError('Provide four finite [x,y] box positions')
    if np.any(p<[.8,.8]) or np.any(p>[19.2,13.2]):raise ValueError('Box centers must stay 0.8 units inside the plane')
    for i in range(4):
        for j in range(i):
            if np.linalg.norm(p[i]-p[j])<1.6-1e-8:raise ValueError('Keep box centers at least 1.6 units apart')
    return p

def random_layout(seed,kind='wide'):
    rng=np.random.default_rng(seed)
    if kind=='original':return np.array([[7.6,7.],[9.2,7.],[10.8,7.],[12.4,7.]])
    if kind in ('rotated','permuted'):
        angle=rng.uniform(-math.pi,math.pi);direction=np.array([math.cos(angle),math.sin(angle)])
        p=rng.uniform([5,5],[15,9])+np.arange(-1.5,2)[:,None]*1.8*direction[None,:]
        if kind=='permuted':p=p[rng.permutation(4)]
        return validate_layout(p)
    low,high=([6,3],[14,11]) if kind=='compact' else ([.8,.8],[19.2,13.2])
    for _ in range(10000):
        p=rng.uniform(low,high,size=(4,2))
        if min(np.linalg.norm(p[i]-p[j]) for i in range(4) for j in range(i))>=1.6:return p
    raise ValueError('Could not generate a valid layout')

def color_sensory(agent,stock=False):
    angles=np.r_[np.linspace(-2.6,-.05,12),np.linspace(.05,2.6,12)]
    colors=np.zeros((4,24),np.float32)
    for i,s in enumerate(agent.stations):
        dx,dy=s['x']-agent.x,s['y']-agent.y;d=math.hypot(dx,dy)
        if d>agent.sensing_range:continue
        bearing=wrap(math.atan2(dy,dx)-agent.heading)
        difference=(angles-bearing+math.pi)%(2*math.pi)-math.pi
        sigma=max(.12,math.atan2(s['radius'],max(d,.1)))
        colors[i]=np.exp(-.5*(difference/sigma)**2)*(1-d/agent.sensing_range)
    cargo=np.asarray([agent.cargo==i for i in (1,2,3)],np.float32)
    sensed=np.r_[agent.sensory(),colors.ravel(),cargo]
    if stock:
        # Visible material inside each box; not a selected destination signal.
        material=colors[:3]*np.asarray([min(s['stock'],3)/3 for s in agent.stations[:3]])[:,None]
        sensed=np.r_[sensed,material.ravel()]
    return sensed.astype(np.float32)

class LayoutSwarmWorld(SwarmWorld):
    def __init__(self,seed=1,flies=4,positions=None,kind='wide',colored=True,stock=False,**kwargs):
        super().__init__(seed=seed,flies=flies,**kwargs)
        self.layout_kind=kind;self.colored=colored;self.stock_sensing=stock
        self.set_layout(random_layout(seed,kind) if positions is None else positions)
        rng=np.random.default_rng(seed+600001)
        for i,a in enumerate(self.agents):
            for _ in range(10000):
                angle=rng.uniform(-math.pi,math.pi);d=rng.uniform(.3,1.5)
                x=self.stations[0]['x']+d*math.cos(angle);y=self.stations[0]['y']+d*math.sin(angle)
                if .22<=x<=19.78 and .22<=y<=13.78 and all(math.hypot(x-b.x,y-b.y)>.48 for b in self.agents[:i]):break
            else:raise ValueError('Could not place flies inside the plane')
            a.x,a.y=x,y;a.trajectory.clear();a.trajectory.append([x,y])
        self.refresh_landmarks()

    def set_layout(self,positions):
        p=validate_layout(positions)
        for s,(x,y) in zip(self.stations,p):s['x'],s['y']=float(x),float(y)
        self.refresh_landmarks()

    def sensory(self):
        self.refresh_landmarks()
        return np.asarray([color_sensory(a,self.stock_sensing) if self.colored else a.sensory() for a in self.agents])

    def snapshot(self,motors):
        value=super().snapshot(motors)
        value['sensory']=self.sensory()[0].tolist()
        value['layout']={'kind':self.layout_kind,'localColorVision':self.colored,
                         'positions':[[s['x'],s['y']] for s in self.stations]}
        return value

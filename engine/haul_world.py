"""Local 2D embodiment; task reward can inspect goals, sensory input cannot.

The action interface and 30-channel projection match the observation baseline.
No route follower, auto-pickup, teleport-to-goal, or action supervision.
"""
from collections import deque
import math
import numpy as np
from .plastic_brain import DT


def wrap(angle): return (angle+math.pi)%(2*math.pi)-math.pi


class HaulWorld:
    width,height,radius,sensing_range=20.,14.,.22,6.
    def __init__(self,seed=1,level=0,horizon=800,continuous=False):
        self.rng=np.random.default_rng(seed); self.level=level; self.horizon=horizon
        self.continuous=continuous
        gap=[1.6,2.5,4.][level]; left=10-gap*1.5
        self.stations=[{'x':left+i*gap,'y':7.,'radius':.55,'kind':name,'color':color,'stock':0,'timer':0.}
            for i,(name,color) in enumerate(zip(('raw ore','smelter','assembler','finished goods'),('#efac69','#e9cc7e','#a3a7ff','#88dec1')))]
        self.stations[0]['stock']=3
        self.landmarks=self.stations; self.obstacles=[]
        angle=self.rng.uniform(-math.pi,math.pi)
        distance=self.rng.uniform(.3,.65 if level==0 else 1.4)
        self.x=left+distance*math.cos(angle); self.y=7+distance*math.sin(angle)
        self.heading=self.rng.uniform(-math.pi,math.pi)
        self.speed=self.turn=self.distance=self.cooldown=0.; self.cargo=0; self.contact=False
        self.collisions=self.deliveries=self.interactions=self.pickups=self.transfers=self.steps=0
        self.total_reward=0.; self.events=deque(maxlen=10); self.trajectory=deque([[self.x,self.y]],maxlen=600)

    def potential(self):
        if self.cargo: targets=[self.stations[self.cargo]]
        else: targets=[s for s in self.stations[:3] if s['stock']>0]
        distance=min((math.hypot(self.x-s['x'],self.y-s['y']) for s in targets),default=0.)
        # Negative potential bounded below, training-only. No target enters act().
        return -.04*min(distance,12.)

    def sensory(self):
        angles=np.r_[np.linspace(-2.6,-.05,12),np.linspace(.05,2.6,12)]
        visual=np.full(24,.025,np.float32)
        for item in self.landmarks:
            dx,dy=item['x']-self.x,item['y']-self.y; d=math.hypot(dx,dy)
            if d>self.sensing_range: continue
            bearing=wrap(math.atan2(dy,dx)-self.heading)
            sigma=max(.12,math.atan2(item['radius'],max(d,.1)))
            difference=(angles-bearing+math.pi)%(2*math.pi)-math.pi
            visual+=np.exp(-.5*(difference/sigma)**2)*(1-d/self.sensing_range)
        smell=[]
        for lateral in (-.2,.2):
            ax=self.x+.25*math.cos(self.heading)-lateral*math.sin(self.heading)
            ay=self.y+.25*math.sin(self.heading)+lateral*math.cos(self.heading)
            smell.append(max((max(0.,1-math.hypot(ax-s['x'],ay-s['y'])/6)**2
                              for s in self.stations[:3] if s['stock']>0),default=0.) if not self.cargo else 0.)
        proximity=max(0.,1-min(self.x,20-self.x,self.y,14-self.y))
        touching=any(s['stock']>0 and math.hypot(self.x-s['x'],self.y-s['y'])<.8 for s in self.stations[:3])
        return np.r_[np.clip(visual,0,1),smell,max(float(self.contact),proximity),
                     float(bool(self.cargo) or touching),abs(self.speed)/2,abs(self.turn)/2].astype(np.float32)

    def advance(self,motor):
        before=self.potential(); event_reward=0.
        self.speed,self.turn=motor['speed'],motor['turn']; self.heading=wrap(self.heading+self.turn*DT)
        x=self.x+self.speed*math.cos(self.heading)*DT; y=self.y+self.speed*math.sin(self.heading)*DT
        blocked=not (self.radius<=x<=20-self.radius and self.radius<=y<=14-self.radius)
        if blocked and not self.contact: self.collisions+=1
        self.contact=blocked
        if not blocked: self.distance+=math.hypot(x-self.x,y-self.y); self.x,self.y=x,y
        self.advance_stations()
        event_reward=self.interact(motor)
        return self.finish_step(before,event_reward,blocked)

    def advance_stations(self):
        for station in self.stations[1:3]:
            if station['timer']>0:
                station['timer']=max(0.,station['timer']-DT)
                if station['timer']==0: station['stock']+=1
    def interact(self,motor):
        event_reward=0.
        self.cooldown=max(0.,self.cooldown-DT)
        if motor['interact'] and self.cooldown<=0:
            self.cooldown=1.; self.interactions+=1
            distances=[math.hypot(self.x-s['x'],self.y-s['y']) for s in self.stations]
            index=int(np.argmin(distances)); station=self.stations[index]
            if distances[index]<.8:
                if self.cargo and index==self.cargo and (index==3 or (station['timer']==0 and station['stock']<2)):
                    self.transfers+=1
                    if index==3: self.deliveries+=1; event_reward=1.; self.events.append('Finished product delivered')
                    else: station['timer']=1. if index==1 else 1.5; event_reward=.2; self.events.append('Material supplied to '+station['kind'])
                    self.cargo=0
                elif not self.cargo and index<3 and station['stock']>0:
                    station['stock']-=1; self.cargo=index+1; self.pickups+=1
                    self.events.append('Picked up from '+station['kind'])
                # Wrong station rejects material. There is no pickup reward to farm.
        return event_reward

    def finish_step(self,before,event_reward,blocked):
        # An external supply replenishes raw stock, not the avatar or its cargo.
        # Default finite training/evaluation worlds retain their original rules.
        if self.continuous: self.stations[0]['stock']=3
        self.steps+=1; terminal=not self.continuous and (self.steps>=self.horizon or self.deliveries>=3)
        after=0. if terminal else self.potential()
        reward=event_reward-.0005-(.002 if blocked else 0.)+.995*after-before
        self.total_reward+=reward; self.trajectory.append([self.x,self.y])
        return reward,terminal

    def snapshot(self,motor):
        return {'avatar':{'x':self.x,'y':self.y,'heading':self.heading,'cargo':bool(self.cargo),'cargoStage':self.cargo,
                    'contact':self.contact,'distance':self.distance,'collisions':self.collisions,'deliveries':self.deliveries,**motor},
                'plane':{'width':20,'height':14,'sensingRange':6,'landmarks':self.stations,'obstacles':[],
                         'trajectory':list(self.trajectory)},'events':list(self.events),'sensory':self.sensory().tolist()}

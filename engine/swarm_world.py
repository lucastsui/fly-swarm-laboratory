"""One shared factory, independent fly bodies, local senses and swept collisions.

Only fly discs and outer walls collide. Stations are collisionless. No routing,
collision avoidance controller, automatic pickup/drop or inter-agent messages.
"""
import math
import numpy as np
from .haul_world import HaulWorld,wrap
from .plastic_brain import DT


def collide(positions,velocities,radius=.22,dt=DT):
    """Equal-mass swept discs, mildly inelastic impacts; no position teleporting.

Earliest time-of-impact prevents tunnelling. Re-solve simultaneous contacts;
if a dense contact chain exhausts the budget, hold the remaining substep.
"""
    p=np.array(positions,dtype=float,copy=True);v=np.array(velocities,dtype=float,copy=True)
    contacts=set();walls=set();remaining=dt;iterations=0
    for iterations in range(128):
        earliest=remaining+1.;hit=None
        for i in range(len(p)):
            for axis,extent in enumerate((20.,14.)):
                speed=v[i,axis]
                if abs(speed)<1e-10:continue
                boundary=extent-radius if speed>0 else radius
                t=(boundary-p[i,axis])/speed
                if -1e-9<=t<=remaining and t<earliest:
                    earliest=max(0.,t);hit=('wall',i,axis)
            for j in range(i):
                delta=p[i]-p[j];relative=v[i]-v[j]
                a=float(relative@relative);b=float(delta@relative)
                c=float(delta@delta)-(2*radius)**2
                if a<1e-14 or b>=-1e-10:continue
                discriminant=b*b-a*c
                if discriminant<0:continue
                t=(-b-math.sqrt(discriminant))/a
                if -1e-8<=t<=remaining and t<earliest:
                    earliest=max(0.,t);hit=('fly',i,j)
        if hit is None:
            p+=v*remaining;remaining=0.;break
        p+=v*earliest;remaining-=earliest
        kind,i,j=hit
        if kind=='wall':
            v[i,j]=0.;walls.add(i)
        else:
            normal=p[i]-p[j];normal/=max(float(np.linalg.norm(normal)),1e-12)
            closing=float((v[i]-v[j])@normal)
            impulse=-.55*closing*normal  # restitution 0.1, equal mass
            v[i]+=impulse;v[j]-=impulse;contacts.add((j,i))
        if remaining<=1e-12:break
    return p,contacts,walls,remaining>1e-9


class SwarmWorld:
    def __init__(self,seed=1,flies=4,continuous=True,horizon=4800,collisions=True):
        if not 1<=flies<=16:raise ValueError('Expected 1–16 flies')
        self.agents=[HaulWorld(seed+i*7919,horizon=horizon,continuous=continuous) for i in range(flies)]
        self.stations=self.agents[0].stations
        self.steps=0;self.collisions_enabled=collisions;self.bump_events=0;self.solver_holds=0
        self.pairs=set();self.continuous=continuous;self.horizon=horizon
        rng=np.random.default_rng(seed+700001)
        for i,agent in enumerate(self.agents):
            agent.stations=self.stations
            if i:
                for _ in range(10000):
                    angle=rng.uniform(-math.pi,math.pi);distance=rng.uniform(.5,1.5)
                    x=self.stations[0]['x']+distance*math.cos(angle);y=7+distance*math.sin(angle)
                    if all(math.hypot(x-a.x,y-a.y)>.48 for a in self.agents[:i]):break
                else:raise ValueError('Could not place non-overlapping flies')
                agent.x,agent.y=x,y
            agent.trajectory.clear();agent.trajectory.append([agent.x,agent.y])
        self.refresh_landmarks()

    def refresh_landmarks(self):
        for agent in self.agents:
            agent.landmarks=self.stations+[{'x':a.x,'y':a.y,'radius':a.radius,'kind':'fly'}
                                           for a in self.agents if a is not agent]

    def sensory(self):
        self.refresh_landmarks()
        return np.asarray([a.sensory() for a in self.agents])

    @property
    def deliveries(self):return sum(a.deliveries for a in self.agents)
    @property
    def pickups(self):return sum(a.pickups for a in self.agents)
    @property
    def transfers(self):return sum(a.transfers for a in self.agents)

    def advance(self,motors):
        if len(motors)!=len(self.agents):raise ValueError('One action per fly is required')
        before=[a.potential() for a in self.agents]
        old=np.asarray([[a.x,a.y] for a in self.agents]);velocities=[]
        for a,m in zip(self.agents,motors):
            a.speed,a.turn=float(m['speed']),float(m['turn'])
            if not (0<=a.speed<=2 and -2<=a.turn<=2):raise ValueError('Motor outside fixed decoder limits')
            a.heading=wrap(a.heading+a.turn*DT)
            velocities.append([a.speed*math.cos(a.heading),a.speed*math.sin(a.heading)])
        if self.collisions_enabled:
            positions,pairs,walls,held=collide(old,velocities)
        else:
            positions=np.clip(old+np.asarray(velocities)*DT,.22,[19.78,13.78])
            pairs=set();walls={i for i,p in enumerate(positions) if np.any(p!=old[i]+np.asarray(velocities[i])*DT)};held=False
        self.bump_events+=len(pairs-self.pairs);self.pairs=pairs;self.solver_holds+=int(held)
        touching={i for pair in pairs for i in pair}|walls
        for i,(a,p) in enumerate(zip(self.agents,positions)):
            if i in touching and not a.contact:a.collisions+=1
            a.contact=i in touching;a.distance+=float(np.linalg.norm(p-old[i]));a.x,a.y=map(float,p)
        # Factory clock advances once, regardless of population. Rotate access
        # ordering so simultaneous requests cannot reserve the same item twice.
        self.agents[0].advance_stations();events=[0.]*len(motors)
        for j in range(len(motors)):
            i=(j+self.steps)%len(motors)
            events[i]=self.agents[i].interact(motors[i])
        rewards=[a.finish_step(b,e,a.contact)[0] for a,b,e in zip(self.agents,before,events)]
        self.steps+=1;self.refresh_landmarks()
        return rewards,not self.continuous and (self.steps>=self.horizon or self.deliveries>=3)

    def snapshot(self,motors):
        result=self.agents[0].snapshot(motors[0])
        result['avatars']=[{'id':i,**a.snapshot(m)['avatar'],'trajectory':list(a.trajectory)}
                           for i,(a,m) in enumerate(zip(self.agents,motors))]
        result['events']=[f'fly {i+1}: {event}' for i,a in enumerate(self.agents) for event in a.events][-16:]
        result['swarm']={'flies':len(self.agents),'collisions':self.collisions_enabled,
                         'bumpEvents':self.bump_events,'solverHolds':self.solver_holds,
                         'sharedWeights':True,'independentNeuralStates':True,'activityFly':1}
        return result

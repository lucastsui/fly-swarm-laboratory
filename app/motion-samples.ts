export type AvatarPose={id:number;x:number;y:number;heading:number;cargoStage:number;contact:boolean;deliveries:number};
export type Pose={step:number;timeMs:number;x:number;y:number;heading:number;cargoStage:number;contact:boolean;deliveries:number;avatars?:AvatarPose[]};
export type MotionPacket={sessionId:string;checkpointHash:string;serverTimeMs:number;running:boolean;samples:Pose[];error?:string};

// Interpolate only between measured physics samples. Never predict beyond the
// latest sample or blend across a service restart. Discrete events stay discrete.
export function interpolatePose(samples:Pose[],time:number):Pose|undefined{
 if(!samples.length)return;
 if(time<=samples[0].timeMs)return samples[0];
 const last=samples[samples.length-1];
 if(time>=last.timeMs)return last;
 for(let i=1;i<samples.length;i++){
  const next=samples[i],previous=samples[i-1];
  if(time>next.timeMs)continue;
  if(time===next.timeMs)return next;
  const fraction=(time-previous.timeMs)/(next.timeMs-previous.timeMs);
  const angle=Math.atan2(Math.sin(next.heading-previous.heading),Math.cos(next.heading-previous.heading));
  const avatars=previous.avatars?.map(a=>{
   const b=next.avatars?.find(b=>b.id===a.id);if(!b)return a;
   const turn=Math.atan2(Math.sin(b.heading-a.heading),Math.cos(b.heading-a.heading));
   return {...a,x:a.x+(b.x-a.x)*fraction,y:a.y+(b.y-a.y)*fraction,heading:a.heading+turn*fraction};
  });
  return {...previous,avatars,x:previous.x+(next.x-previous.x)*fraction,
   y:previous.y+(next.y-previous.y)*fraction,heading:previous.heading+angle*fraction};
 }
 return last;
}

export class MotionPlayback{
 packet:MotionPacket|null=null;
 received=0;
 renderedTime=-Infinity;
 displayed:Pose|undefined;
 bufferMs=280;
 underruns=0;
 private lastFrame:number|undefined;
 private arrivals:number[]=[];
 accept(packet:MotionPacket,now:number){
  if(!packet.samples.length)return false;
  if(this.packet?.sessionId!==packet.sessionId){
   this.displayed=undefined;this.renderedTime=-Infinity;this.lastFrame=undefined;
   this.arrivals=[];this.bufferMs=280;this.underruns=0;
  }else{
   if(packet.samples.at(-1)!.step<this.packet.samples.at(-1)!.step)return false;
   this.arrivals.push(now-this.received);this.arrivals=this.arrivals.slice(-40);
   const gaps=[...this.arrivals].sort((a,b)=>a-b);
   const desired=Math.min(750,Math.max(240,gaps[Math.floor((gaps.length-1)*.95)]*1.5+80));
   this.bufferMs+=(desired-this.bufferMs)*(desired>this.bufferMs ? .3 : .02);
   // Incremental responses retain the already-buffered measured history.
   const samples=new Map(this.packet.samples.map(p=>[p.step,p]));
   for(const sample of packet.samples)samples.set(sample.step,sample);
   packet={...packet,samples:[...samples.values()].sort((a,b)=>a.step-b.step).slice(-512)};
  }
  this.packet=packet;this.received=now;
  return true;
 }
 frame(now:number,enabled=true){
  if(!this.packet)return this.displayed;
  const elapsed=this.lastFrame===undefined?0:Math.max(0,now-this.lastFrame);
  this.lastFrame=now;
  if(!enabled||!this.packet.running||this.packet.error||now-this.received>1500){
   return this.displayed??this.packet.samples.at(-1);
  }
  const first=this.packet.samples[0],last=this.packet.samples.at(-1)!;
  const target=this.packet.serverTimeMs+now-this.received-this.bufferMs;
  if(!Number.isFinite(this.renderedTime))this.renderedTime=Math.max(first.timeMs,Math.min(last.timeMs,target));
  else{
   // Advance one continuous playback clock. Packet arrivals only gently steer
   // its rate; they never reset it or cause an immediate catch-up jump.
   const correction=Math.max(-.15,Math.min(.08,(target-this.renderedTime)/1500));
   const headroom=Math.max(0,last.timeMs-this.renderedTime);
   const rate=(1+correction)*Math.min(1,headroom/60);
   const next=this.renderedTime+Math.min(elapsed,80)*rate;
   if(headroom<2&&elapsed>0)this.underruns++;
   this.renderedTime=Math.min(last.timeMs,Math.max(first.timeMs,next));
  }
  this.displayed=interpolatePose(this.packet.samples,this.renderedTime);
  return this.displayed;
 }
}

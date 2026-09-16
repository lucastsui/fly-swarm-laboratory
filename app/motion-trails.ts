import type {AvatarPose,MotionPacket} from "./motion-samples";

type Point={step:number;x:number;y:number;svg?:string};
export type TrailSeed={sessionId:string;step:number;avatars:{id:number;trajectory:number[][]}[]};

// Retain measured physics positions, not a frame-rate-dependent trail. Extra
// capacity holds the motion buffer's future samples without shortening the tail.
export class MotionTrails{
 sessionId="";
 seeded=false;
 points=new Map<number,Point[]>();
 merge(id:number,incoming:Point[]){
  const byStep=new Map((this.points.get(id)??[]).map(p=>[p.step,p]));
  for(const p of incoming)byStep.set(p.step,p);
  this.points.set(id,[...byStep.values()].sort((a,b)=>a.step-b.step).slice(-632));
 }
 accept(packet:MotionPacket){
  if(packet.sessionId!==this.sessionId){this.sessionId=packet.sessionId;this.seeded=false;this.points.clear();}
  const incoming=new Map<number,Point[]>();
  for(const sample of packet.samples)for(const a of sample.avatars??[{...sample,id:0}]){
   const points=incoming.get(a.id)??[];
   points.push({step:sample.step,x:a.x,y:a.y});incoming.set(a.id,points);
  }
  for(const [id,points] of incoming)this.merge(id,points);
 }
 seed(seed?:TrailSeed){
  if(!seed||this.seeded||seed.sessionId!==this.sessionId)return;
  for(const a of seed.avatars){
   const history=a.trajectory.map(([x,y],i)=>({step:seed.step-a.trajectory.length+1+i,x,y}));
   // Prefer the high-frequency stream when a seed and packet overlap.
   const measured=this.points.get(a.id)??[];
   this.points.set(a.id,history);this.merge(a.id,measured);
  }
  this.seeded=true;
 }
 path(avatar:AvatarPose,step:number){
  const history=(this.points.get(avatar.id)??[]).filter(p=>p.step<=step).slice(-599);
  let previous:Point|undefined;
  const pieces=history.map(p=>{
   const command=!previous||p.step>previous.step+1?"M":"L";
   previous=p;p.svg??=(p.x*40).toFixed(3)+" "+(p.y*40).toFixed(3);return command+p.svg;
  });
  // This is exactly the interpolated avatar position, never a future sample.
  const command=!previous||step>previous.step+1?"M":"L";
  pieces.push(command+(avatar.x*40).toFixed(3)+" "+(avatar.y*40).toFixed(3));
  return pieces.join(" ");
 }
}

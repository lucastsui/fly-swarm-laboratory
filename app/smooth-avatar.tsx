"use client";
import {useEffect,useRef,useState} from "react";
import {MotionPlayback,type MotionPacket} from "./motion-samples";
import {MotionTrails,type TrailSeed} from "./motion-trails";

export const FLY_COLORS=["#c0f3df","#82c7ff","#f3b3ec","#ffd080"];
export default function SmoothAvatar({api,checkpointHash,sensingRange,flies=1,trailSeed}:{api:string;checkpointHash:string;sensingRange:number;flies?:number;trailSeed?:TrailSeed}){
 const position=useRef<(SVGGElement|null)[]>([]),rotation=useRef<(SVGGElement|null)[]>([]),body=useRef<(SVGCircleElement|null)[]>([]),cargo=useRef<(SVGCircleElement|null)[]>([]);
 const trails=useRef<(SVGPathElement|null)[]>([]),initialTrail=useRef(trailSeed);initialTrail.current=trailSeed;
 const diagnostics=useRef<SVGGElement|null>(null);
 const [status,setStatus]=useState("Connecting to motion…");
 useEffect(()=>{
  const playback=new MotionPlayback(),history=new MotionTrails();let active=true,frameId=0;
  let previousPose:ReturnType<MotionPlayback['frame']>;
  let frameCount=0,lastFrame=0,maxFrameGap=0,lastDiagnostics=0;
  let timer:ReturnType<typeof setTimeout>,controller:AbortController|undefined;
  const poll=async()=>{
   const began=performance.now();controller=new AbortController();
   const timeout=setTimeout(()=>controller?.abort(),1500);
   try{
    const latest=playback.packet;
    const query=latest?`?after=${latest.samples.at(-1)!.step}&session=${encodeURIComponent(latest.sessionId)}`:"";
    const response=await fetch(api+"/motion"+query,{cache:"no-store",signal:controller.signal});
    if(!response.ok)throw new Error("Motion unavailable");
    const packet=await response.json() as MotionPacket;
    if(packet.checkpointHash!==checkpointHash||!packet.samples?.length)throw new Error("Motion/checkpoint mismatch");
    if(!active)return;
    if(playback.accept(packet,performance.now())){history.accept(packet);history.seed(initialTrail.current);}
    setStatus(packet.error?"Motion paused: "+packet.error:"");
   }catch(error){if(active)setStatus("Motion feed unavailable; position held.");}
   finally{clearTimeout(timeout);if(active)timer=setTimeout(poll,Math.max(10,50-(performance.now()-began)));}
  };
  const animate=(now:number)=>{
   // The movement feed owns its running/stale state. A delayed statistics
   // request must not freeze otherwise healthy avatar animation.
   const pose=playback.frame(now);
   frameCount++;if(lastFrame)maxFrameGap=Math.max(maxFrameGap,now-lastFrame);lastFrame=now;
   if(now-lastDiagnostics>500&&diagnostics.current){
    const node=diagnostics.current;lastDiagnostics=now;
    node.dataset.frames=String(frameCount);node.dataset.maxFrameGapMs=maxFrameGap.toFixed(1);
    node.dataset.bufferMs=playback.bufferMs.toFixed(0);node.dataset.underruns=String(playback.underruns);
    node.dataset.step=String(pose?.step??0);
    node.dataset.feedAgeMs=(now-playback.received).toFixed(0);
   }
   if(pose&&pose!==previousPose)for(const a of pose.avatars??[{...pose,id:0}]){
    trails.current[a.id]?.setAttribute("d",history.path(a,pose.step));
   }
   previousPose=pose;
   if(pose)for(const a of pose.avatars??[{...pose,id:0}]){
    const i=a.id;
    position.current[i]?.setAttribute("transform",`translate(${a.x*40},${a.y*40})`);
    rotation.current[i]?.setAttribute("transform",`rotate(${a.heading*180/Math.PI})`);
    body.current[i]?.setAttribute("fill",a.contact?"#f48362":FLY_COLORS[i%FLY_COLORS.length]);
    body.current[i]?.setAttribute("stroke-width",a.contact?"3":"1.5");
    cargo.current[i]?.setAttribute("visibility",a.cargoStage?"visible":"hidden");
    cargo.current[i]?.setAttribute("fill",["","#efac69","#e9cc7e","#a3a7ff"][a.cargoStage]||"#efac69");
    position.current[i]?.setAttribute("visibility","visible");
   }
   frameId=requestAnimationFrame(animate);
  };
  void poll();frameId=requestAnimationFrame(animate);
  return()=>{active=false;clearTimeout(timer);controller?.abort();cancelAnimationFrame(frameId);};
 },[api,checkpointHash]);
 return <g pointerEvents="none" ref={diagnostics} data-testid="motion-playback" data-playback-version="continuous-clock-v2">
  {Array.from({length:flies},(_,i)=><path key={i} ref={el=>{trails.current[i]=el}} data-testid="smooth-trail" data-fly-id={i+1} fill="none" stroke={FLY_COLORS[i%4]+"70"} strokeWidth="1.4" strokeLinejoin="round" strokeLinecap="round"/>)}
  {Array.from({length:flies},(_,i)=><g key={i} ref={el=>{position.current[i]=el}} visibility="hidden" data-testid="smooth-avatar" data-fly-id={i+1}>
   {i===0&&<circle r={sensingRange*40} fill="#88dec106" stroke="#88dec12b" strokeDasharray="4 7"/>}
   <circle ref={el=>{body.current[i]=el}} r="8.8" fill={FLY_COLORS[i%4]} stroke="#07130e" strokeWidth="1.5"/>
   <g ref={el=>{rotation.current[i]=el}}><path d="M3 -4L8 0L3 4M9 0H19" fill="none" stroke="#356451" strokeWidth="1.5"/></g>
   <circle ref={el=>{cargo.current[i]=el}} cx="-2" cy="0" r="3.5" stroke="#142319" fill="#efac69" visibility="hidden"/>
   <text y="-15" textAnchor="middle" fill={FLY_COLORS[i%4]} fontSize="12">{i+1}</text>
  </g>)}
  {status&&<text x="24" y="535" className="map-meta">{status}</text>}
 </g>;
}

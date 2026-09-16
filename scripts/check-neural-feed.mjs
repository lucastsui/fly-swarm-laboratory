import fs from 'node:fs';
import {performance} from 'node:perf_hooks';
import {decodeNeuralSample} from '../app/neural-stream.ts';
const base='http://127.0.0.1:8770/api/plane';
const metadata=await (await fetch(base+'/neural/metadata')).json();
const latencies=[],motionLatency=[],steps=new Set();let changed=0,prior,hash='',session='',firstTime,lastTime,maxDifference=0;
const start=performance.now();
while(performance.now()-start<20000){
 const began=performance.now();
 const response=await fetch(base+'/neural/activity');
 if(!response.ok)throw Error('Neural endpoint failed');
 const sample=decodeNeuralSample(await response.arrayBuffer());
 latencies.push(performance.now()-began);steps.add(sample.step);
 hash=response.headers.get('x-brain-checkpoint');session=response.headers.get('x-brain-session');
 if(hash!==metadata.checkpointHash)throw Error('Mismatched model');
 firstTime??=sample.timeMs;lastTime=sample.timeMs;
 if(prior){
  let differs=false;
  for(let i=0;i<metadata.connectionCount;i++){
   const source=metadata.source[i];const diff=Math.abs(metadata.weight[i]*(sample.source[source]-prior.source[source]));
   if(diff>0)differs=true;maxDifference=Math.max(diff,maxDifference);
  }
  if(differs)changed++;
 }
 prior=sample;
 if(latencies.length%5===0){
  const at=performance.now();const motion=await fetch(base+'/motion?after='+sample.step+'&session='+encodeURIComponent(session));await motion.arrayBuffer();motionLatency.push(performance.now()-at);
 }
 await new Promise(resolve=>setTimeout(resolve,Math.max(1,50-(performance.now()-began))));
}
const stats=values=>{values.sort((a,b)=>a-b);return {median:values[Math.floor(values.length*.5)],p95:values[Math.floor(values.length*.95)],max:values.at(-1)}};
const report={checkpointHash:hash,session,connections:metadata.connectionCount,totalConnections:metadata.totalConnections,packets:latencies.length,uniqueNeuralSamples:steps.size,observedSampleHz:(steps.size-1)/((lastTime-firstTime)/1000),changedPackets:changed,maxConnectionDifference:maxDifference,neuralLatencyMs:stats(latencies),motionLatencyMs:stats(motionLatency),packetBytes:24+metadata.neuronCount*8,definition:metadata.definition};
if(process.argv[2])fs.writeFileSync(process.argv[2],JSON.stringify(report,null,2));
console.log(JSON.stringify(report,null,2));

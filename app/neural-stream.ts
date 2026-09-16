/** Sampled model measurements; no generated spikes, motion or propagation. */
export type NeuralMetadata={schema:string;checkpointHash:string;nodes:{id:number}[];source:number[];target:number[];edgeIndex:number[];weight:number[];connectionCount:number;totalConnections:number;neuronCount:number;totalNeurons:number;fly:number};
export type NeuralSample={step:number;timeMs:number;activity:Float32Array;source:Float32Array};
export function decodeNeuralSample(bytes:ArrayBuffer):NeuralSample{
 const header=new DataView(bytes);
 if(bytes.byteLength<24||header.getUint32(0,true)!==0x31425346)throw Error('Invalid neural sample');
 const count=header.getUint32(20,true);
 if(count>200000||bytes.byteLength!==24+count*8)throw Error('Incomplete neural sample');
 const activity=new Float32Array(count),source=new Float32Array(count);
 for(let i=0;i<count;i++){
  activity[i]=header.getFloat32(24+i*4,true);source[i]=header.getFloat32(24+(count+i)*4,true);
  if(!Number.isFinite(activity[i])||!Number.isFinite(source[i]))throw Error('Non-finite neural measurement');
 }
 return {step:header.getFloat64(4,true),timeMs:header.getFloat64(12,true),activity,source};
}

export class NeuralPlayback{
 samples:NeuralSample[]=[];received=0;session='';running=false;count=0;
 private rendered=-Infinity;private frameTime:number|undefined;
 accept(sample:NeuralSample,session:string,running:boolean,now:number){
  if(session!==this.session){this.samples=[];this.rendered=-Infinity;this.frameTime=undefined;this.count=0;}
  this.session=session;this.running=running;this.received=now;
  const last=this.samples.at(-1);
  if(last&&sample.step<=last.step)return;
  this.samples.push(sample);this.samples=this.samples.slice(-48);this.count++;
 }
 frame(now:number){
  const last=this.samples.at(-1);if(!last)return null;
  const elapsed=this.frameTime===undefined?0:Math.max(0,now-this.frameTime);this.frameTime=now;
  const live=this.running&&now-this.received<1500;
  if(!Number.isFinite(this.rendered))this.rendered=Math.max(this.samples[0].timeMs,last.timeMs-160);
  else if(live){
   const target=last.timeMs+now-this.received-160;
   const rate=1+Math.max(-.15,Math.min(.10,(target-this.rendered)/1200));
   this.rendered=Math.min(last.timeMs,Math.max(this.samples[0].timeMs,this.rendered+Math.min(elapsed,80)*rate));
  }
  let a=this.samples[0],b=last;
  for(const sample of this.samples){if(sample.timeMs<=this.rendered)a=sample;else{b=sample;break;}}
  const fraction=b.timeMs>a.timeMs?Math.max(0,Math.min(1,(this.rendered-a.timeMs)/(b.timeMs-a.timeMs))):0;
  return {a,b,fraction,live,step:a.step,ageMs:now-this.received};
 }
}

export class NeuralStream{
 metadata:NeuralMetadata|null=null;playback=new NeuralPlayback();error='Connecting to neural measurements';
 private active=false;private timer:ReturnType<typeof setTimeout>|undefined;private controller:AbortController|undefined;
 readonly api:string;readonly checkpointHash:string;
 constructor(api:string,checkpointHash:string){this.api=api;this.checkpointHash=checkpointHash;}
 start(){this.active=true;void this.poll();}
 stop(){this.active=false;clearTimeout(this.timer);this.controller?.abort();}
 private async poll(){
  const began=performance.now();this.controller=new AbortController();const timeout=setTimeout(()=>this.controller?.abort(),1500);
  try{
   if(!this.metadata){
    const response=await fetch(this.api+'/neural/metadata',{cache:'no-store',signal:this.controller.signal});
    if(!response.ok)throw Error('High-rate connection measurements unavailable');
    const metadata=await response.json() as NeuralMetadata;
    if(metadata.schema!=='connection-signals-v1'||metadata.checkpointHash!==this.checkpointHash||metadata.nodes.length!==metadata.neuronCount||[metadata.source,metadata.target,metadata.weight,metadata.edgeIndex].some(a=>a.length!==metadata.connectionCount))throw Error('Neural topology/checkpoint mismatch');
    this.metadata=metadata;
   }
   const response=await fetch(this.api+'/neural/activity',{cache:'no-store',signal:this.controller.signal});
   if(!response.ok)throw Error('Neural measurements unavailable');
   if(response.headers.get('X-Brain-Checkpoint')!==this.checkpointHash)throw Error('Neural sample/checkpoint mismatch');
   const session=response.headers.get('X-Brain-Session');if(!session)throw Error('Missing neural session');
   const sample=decodeNeuralSample(await response.arrayBuffer());
   if(sample.activity.length!==this.metadata.neuronCount)throw Error('Neural sample/topology mismatch');
   if(!this.active)return;
   this.playback.accept(sample,session,response.headers.get('X-Brain-Running')==='1',performance.now());this.error='';
  }catch(error){if(this.active)this.error=error instanceof Error?error.message:'Neural stream unavailable';}
  finally{clearTimeout(timeout);if(this.active)this.timer=setTimeout(()=>void this.poll(),this.error?1000:Math.max(10,50-(performance.now()-began)));}
 }
}

export function weightedConnectionSignal(weight:number,source:number){return weight*source;}

/** Display gain on an actual sample-to-sample change, not a synthetic spike. */
export function measuredActivityChange(previous:number,current:number){
 if(!Number.isFinite(previous)||!Number.isFinite(current))return 0;
 const difference=Math.abs(current-previous);
 const scale=Math.max(Math.abs(previous),Math.abs(current));
 const threshold=Math.max(.00002,scale*.0025);
 return difference<=threshold?0:Math.sqrt(Math.min(1,(difference-threshold)/Math.max(.0003,scale*.05)));
}

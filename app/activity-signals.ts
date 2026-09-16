/** Measured activation → display values. No synthetic events or spike model. */
export const ACTIVITY_LOG_GAIN = 5000;
export const CHANGE_HIGHLIGHT_MS = 650;
export const activityLevel = (value:number) => Math.min(1, Math.log1p(Math.abs(value)*ACTIVITY_LOG_GAIN)/Math.log1p(ACTIVITY_LOG_GAIN));

export type ActivitySample = {nodes:readonly {id:number}[];activity:readonly number[]};
export type SampleContext = {step?:number;runId?:string;running:boolean;connected:boolean};
export type ActivityStats = {step:number|null;count:number;highlighted:number;maxChange:number;status:"live"|"paused"|"offline"};

export class ActivitySignals {
  /** RGBA per neuron: log magnitude, highlight strength, event time, observed. */
  readonly pixels:Float32Array;
  private targets:Float32Array;
  private previous:Float64Array;
  private lookup:Map<number,number>;
  private lastStep:number|undefined;
  private lastRun:string|undefined;
  private lastSampleTime=-Infinity;
  private baseline=false;
  stats:ActivityStats={step:null,count:0,highlighted:0,maxChange:0,status:"offline"};

  constructor(ids:readonly number[],textureWidth:number){
    this.pixels=new Float32Array(textureWidth*4);
    this.targets=new Float32Array(textureWidth);
    this.previous=new Float64Array(textureWidth).fill(NaN);
    this.lookup=new Map(ids.map((id,index)=>[id,index]));
  }

  accept(sample:ActivitySample|undefined,context:SampleContext,now:number):boolean{
    const previousStatus=this.stats.status;
    const status=!context.connected||!sample?"offline":context.running?"live":"paused";
    this.stats={...this.stats,status};
    if(status==="offline"){
      this.clearHighlights();this.baseline=false;this.previous.fill(NaN);
      return previousStatus!==status;
    }
    if(!sample)return false;
    if(!context.running)this.clearHighlights();
    const sameRun=context.runId===this.lastRun;
    if(this.baseline&&sameRun&&context.step!==undefined&&context.step===this.lastStep){
      // A paused engine or repeated HTTP snapshot must never retrigger a pulse.
      return previousStatus!==status;
    }
    const continuous=this.baseline&&sameRun&&now-this.lastSampleTime<3000&&
      (context.step===undefined||this.lastStep===undefined||context.step>this.lastStep);
    const seen=new Set<number>();
    let count=0,highlighted=0,maxChange=0;
    sample.nodes.forEach((node,index)=>{
      const owner=this.lookup.get(node.id),value=sample.activity[index];
      if(owner===undefined||!Number.isFinite(value)||seen.has(owner))return;
      seen.add(owner);count++;
      const offset=owner*4,previous=this.previous[owner];
      this.targets[owner]=activityLevel(value);
      if(!continuous||!Number.isFinite(previous))this.pixels[offset]=this.targets[owner];
      this.pixels[offset+1]=0;
      this.pixels[offset+3]=1;
      if(continuous&&context.running&&Number.isFinite(previous)){
        const change=Math.abs(value-previous);
        maxChange=Math.max(maxChange,change);
        // Ignore numerical-scale jitter; amplify small, real sample changes.
        const threshold=Math.max(.00002,Math.abs(previous)*.0025);
        if(change>threshold){
          this.pixels[offset+1]=Math.sqrt(Math.min(1,(change-threshold)/Math.max(.0003,Math.abs(previous)*.08)));
          this.pixels[offset+2]=now/1000;
          highlighted++;
        }
      }
      this.previous[owner]=value;
    });
    for(let owner=0;owner<this.targets.length;owner++)if(!seen.has(owner)){
      this.pixels[owner*4+3]=0;this.pixels[owner*4+1]=0;this.previous[owner]=NaN;
    }
    this.lastStep=context.step;this.lastRun=context.runId;this.lastSampleTime=now;this.baseline=true;
    this.stats={step:context.step??null,count,highlighted,maxChange,status};
    return true;
  }

  clearHighlights(){
    for(let i=1;i<this.pixels.length;i+=4)this.pixels[i]=0;
    this.stats={...this.stats,highlighted:0};
  }

  advance(fraction:number){
    for(let owner=0;owner<this.targets.length;owner++){
      const index=owner*4;
      this.pixels[index]+=(this.targets[owner]-this.pixels[index])*fraction;
    }
  }
}

import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
import type {BrainView} from "./brain-activity";
import {ActivitySignals,CHANGE_HIGHLIGHT_MS,type SampleContext,type ActivityStats} from "./activity-signals";
import {activityLevel} from "./activity-signals";
import {measuredActivityChange,type NeuralStream,type NeuralMetadata} from "./neural-stream";

export type Neuron = {id:number;type:string;superclass:string;side:string;soma:[number,number,number]|null;live:boolean;vertices:number;segments:number};
export type Manifest = {
  dataset:string;neurons:Neuron[];neuronCount:number;liveNeuronCount:number;
  vertexCount:number;segmentCount:number;compressedBytes:number;
  bounds:[number[],number[]];
  chunks:{url:string;vertices:number;segments:number;bytes:number;sha256:string}[];
};
type Chunk={positions:Float32Array;owners:Uint32Array;indices:Uint32Array};
export type Morphology={manifest:Manifest;chunks:Chunk[]};
export type Preset="Oblique"|"Front"|"Side";
export type DisplayMode="Anatomy"|"Activity";

let cached:Promise<Morphology>|null=null;
export function loadMorphology(progress:(value:string)=>void):Promise<Morphology>{
  if(cached)return cached;
  cached=(async()=>{
    const response=await fetch("/anatomy/manifest.json");
    if(!response.ok)throw new Error("The morphology manifest could not be loaded.");
    const manifest=await response.json() as Manifest;
    const chunks:Chunk[]=new Array(manifest.chunks.length);
    let completed=0,next=0;
    await Promise.all(Array.from({length:3},async()=>{
      while(next<manifest.chunks.length){
        const index=next++,part=manifest.chunks[index];
        const fetched=await fetch(part.url);
        if(!fetched.ok)throw new Error("A neuron geometry file could not be loaded.");
        const raw=await fetched.arrayBuffer();
        const bytes=new Uint8Array(raw);
        const binary=bytes[0]===31&&bytes[1]===139
          ?await new Response(new Blob([raw]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer()
          :raw;
        const header=new DataView(binary);
        if(header.getUint32(0,true)!==0x534e434d||header.getUint32(4,true)!==1)throw new Error("Unsupported neuron geometry format.");
        const vertices=header.getUint32(8,true),segments=header.getUint32(12,true);
        if(vertices!==part.vertices||segments!==part.segments||binary.byteLength!==16+vertices*16+segments*8)throw new Error("Incomplete neuron geometry.");
        chunks[index]={positions:new Float32Array(binary,16,vertices*3),owners:new Uint32Array(binary,16+vertices*12,vertices),indices:new Uint32Array(binary,16+vertices*16,segments*2)};
        completed++;
        progress(`Loading traced branches · ${Math.round(completed/manifest.chunks.length*100)}%`);
      }
    }));
    return {manifest,chunks};
  })().catch(error=>{cached=null;throw error});
  return cached;
}

const vertexShader=`
attribute float neuron;
uniform sampler2D activityMap;
uniform float activityWidth;
uniform float displayMode;
uniform float pointScale;
uniform float sampleTime;
uniform float highlightEnabled;
uniform float reducedMotion;
varying vec3 cellColor;
varying float cellAlpha;
varying float cellPulse;
vec3 palette(float n) {
  float hue=fract(n*.61803398875);
  vec3 p=abs(fract(vec3(hue)+vec3(0.,.6666667,.3333333))*6.-3.);
  return mix(vec3(.38),clamp(p-1.,0.,1.),.62)*.95;
}
void main(){
  vec4 reading=texture2D(activityMap,vec2((neuron+.5)/activityWidth,.5));
  cellColor=palette(neuron);
  cellAlpha=.48;
  cellPulse=0.;
  if(displayMode>.5){
    float level=reading.r;
    if(level<.25)cellColor=mix(vec3(.09,.16,.36),vec3(.35,.35,1.),level*4.);
    else if(level<.5)cellColor=mix(vec3(.35,.35,1.),vec3(0.,.8,.88),(level-.25)*4.);
    else if(level<.75)cellColor=mix(vec3(0.,.8,.88),vec3(.45,.97,.57),(level-.5)*4.);
    else cellColor=mix(vec3(.45,.97,.57),vec3(1.,.65,.25),(level-.75)*4.);
    cellPulse=reading.g*highlightEnabled*.7;
    cellColor=mix(cellColor,vec3(1.,.94,.60),cellPulse);
    cellAlpha=.10+level*.48+cellPulse*.65;
    if(reading.a<.5){cellColor=vec3(.35,.45,.55);cellAlpha=.004;cellPulse=0.;}
  }
  gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);
  gl_PointSize=clamp(pointScale*3.6+cellPulse*10.,1.6,30.);
}`;
const fragmentShader=`
varying vec3 cellColor;
varying float cellAlpha;
void main(){gl_FragColor=vec4(cellColor,cellAlpha);}`;
const somaFragment=`
varying vec3 cellColor;
varying float cellAlpha;
varying float cellPulse;
void main(){
  vec2 p=gl_PointCoord*2.-1.;
  float r=dot(p,p);
  if(r>1.)discard;
  float lighting=.78+.30*sqrt(1.-r)-.10*p.x-.10*p.y;
  float halo=mix(1.,.25+.75*pow(1.-r,1.3),min(1.,cellPulse*2.));
  gl_FragColor=vec4(cellColor*lighting,min(1.,cellAlpha*1.8)*halo);
}`;

export class MorphologyRenderer{
  private renderer:THREE.WebGLRenderer;
  private scene=new THREE.Scene();
  private camera=new THREE.OrthographicCamera(-1,1,1,-1,.1,10000);
  private controls:OrbitControls;
  private group=new THREE.Group();
  private geometries:THREE.BufferGeometry[]=[];
  private materials:THREE.ShaderMaterial[]=[];
  private signals:ActivitySignals;
  private texture:THREE.DataTexture;
  private uniforms;
  private frame=0;
  private dirty=true;
  private active=true;
  private visible=true;
  private disposed=false;
  private rotation=false;
  private mode:DisplayMode="Anatomy";
  private smoothingUntil=0;
  private changesUntil=0;
  private settling=false;
  private lastRenderedTime=0;
  private reducedMotion=window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  private lastFrame=0;
  private observer:ResizeObserver;
  private visibilityObserver:IntersectionObserver;
  private viewHeight=1200;
  private preset:Preset="Oblique";
  private bounds:THREE.Vector3[];
  private neuralStream:NeuralStream|null=null;
  private neuralMetadata:NeuralMetadata|null=null;
  private neuralOwners:number[]=[];
  private renderedFrames=0;

  constructor(private host:HTMLElement,private data:Morphology,private onScale:(pixels:number)=>void){
    this.renderer=new THREE.WebGLRenderer({antialias:true,alpha:false,powerPreference:"low-power"});
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,1.5));
    this.renderer.setClearColor(0xffffff);
    const canvas=this.renderer.domElement;
    canvas.setAttribute("role","img");
    canvas.setAttribute("aria-label",`Interactive 3D MaleCNS anatomy: ${data.manifest.neuronCount} real neuron skeletons. Original traced branches preserve spatial curvature. Drag to rotate or use the view buttons.`);
    canvas.tabIndex=0;
    this.host.appendChild(canvas);
    const count=THREE.MathUtils.ceilPowerOfTwo(data.manifest.neuronCount);
    this.signals=new ActivitySignals(data.manifest.neurons.map(n=>n.id),count);
    this.texture=new THREE.DataTexture(this.signals.pixels,count,1,THREE.RGBAFormat,THREE.FloatType);
    this.texture.needsUpdate=true;
    this.uniforms={activityMap:{value:this.texture},activityWidth:{value:count},displayMode:{value:0},pointScale:{value:1},sampleTime:{value:0},highlightEnabled:{value:0},reducedMotion:{value:this.reducedMotion?1:0}};
    const lineMaterial=new THREE.ShaderMaterial({vertexShader,fragmentShader,uniforms:this.uniforms,transparent:true,depthWrite:false});
    const somaMaterial=new THREE.ShaderMaterial({vertexShader,fragmentShader:somaFragment,uniforms:this.uniforms,transparent:true,depthWrite:false});
    this.materials.push(lineMaterial,somaMaterial);
    for(const chunk of data.chunks){
      const geometry=new THREE.BufferGeometry();
      geometry.setAttribute("position",new THREE.BufferAttribute(chunk.positions,3));
      geometry.setAttribute("neuron",new THREE.Float32BufferAttribute(chunk.owners,1));
      geometry.setIndex(new THREE.BufferAttribute(chunk.indices,1));
      geometry.computeBoundingSphere();
      this.geometries.push(geometry);
      this.group.add(new THREE.LineSegments(geometry,lineMaterial));
    }
    const somas:number[]=[],somaIds:number[]=[];
    data.manifest.neurons.forEach((n,i)=>{if(n.soma){somas.push(...n.soma);somaIds.push(i)}});
    const somaGeometry=new THREE.BufferGeometry();
    somaGeometry.setAttribute("position",new THREE.Float32BufferAttribute(somas,3));
    somaGeometry.setAttribute("neuron",new THREE.Float32BufferAttribute(somaIds,1));
    this.geometries.push(somaGeometry);
    this.group.add(new THREE.Points(somaGeometry,somaMaterial));
    const [lo,hi]=data.manifest.bounds;
    const center=lo.map((v,i)=>(v+hi[i])/2);
    // A single rigid rotation: no separate bending or repositioning of the VNC.
    this.group.rotation.x=Math.PI/2;
    this.group.position.set(-center[0],center[2],-center[1]);
    this.scene.add(this.group);
    this.bounds=[];
    // Fit to a dense sample of the actual anatomy, not the empty corners of
    // its bounding box. The geometry itself remains in the source space.
    for(const chunk of data.chunks)for(let i=0;i<chunk.positions.length;i+=3*64){
      const p=chunk.positions;
      this.bounds.push(new THREE.Vector3(p[i]-center[0],center[2]-p[i+2],p[i+1]-center[1]));
    }
    this.controls=this.createControls();
    canvas.addEventListener("keydown",this.onKey);
    canvas.addEventListener("webglcontextlost",this.onContextLost);
    this.observer=new ResizeObserver(()=>this.resize());
    this.observer.observe(host);
    this.visibilityObserver=new IntersectionObserver(entries=>{this.visible=entries[0]?.isIntersecting??true;this.dirty=true});
    this.visibilityObserver.observe(host);
    this.setPreset("Oblique");
    this.resize();
    this.frame=requestAnimationFrame(this.animate);
  }
  private onContextLost=(event:Event)=>{
    event.preventDefault();
    const message=document.createElement("div");
    message.className="morphology-error";
    message.textContent="The 3D graphics context was interrupted. Reload the page to restore the view.";
    this.host.appendChild(message);
    this.active=false;
  };
  private onKey=(event:KeyboardEvent)=>{
    if(event.key==="+"||event.key==="="){event.preventDefault();this.zoom(1.25)}
    if(event.key==="-"){event.preventDefault();this.zoom(.8)}
    if(event.key.toLowerCase()==="r"){event.preventDefault();this.setPreset(this.preset)}
  };
  private createControls(){
    const controls=new OrbitControls(this.camera,this.renderer.domElement);
    controls.enableDamping=true;controls.dampingFactor=.12;
    controls.autoRotateSpeed=.7;controls.autoRotate=this.rotation;
    controls.minZoom=.6;controls.maxZoom=12;
    controls.addEventListener("change",()=>{this.dirty=true});
    return controls;
  }
  private resize(){
    const {width,height}=this.host.getBoundingClientRect();
    if(!width||!height)return;
    this.renderer.setSize(width,height,false);
    const aspect=width/height;
    this.camera.left=-this.viewHeight*aspect/2;
    this.camera.right=this.viewHeight*aspect/2;
    this.camera.top=this.viewHeight/2;
    this.camera.bottom=-this.viewHeight/2;
    this.camera.updateProjectionMatrix();
    this.dirty=true;
  }
  setPreset(preset:Preset){
    this.preset=preset;
    this.controls.dispose();
    this.camera.zoom=1;
    this.camera.up.set(preset==="Oblique"?-2:0,1,0).normalize();
    this.camera.position.copy(preset==="Front"?new THREE.Vector3(0,0,2200):preset==="Side"?new THREE.Vector3(-2200,0,0):new THREE.Vector3(-2000,500,1000));
    this.camera.lookAt(0,0,0);
    this.camera.updateMatrixWorld();
    let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
    const projected=new THREE.Vector3();
    for(const point of this.bounds){
      projected.copy(point).applyMatrix4(this.camera.matrixWorldInverse);
      minX=Math.min(minX,projected.x);maxX=Math.max(maxX,projected.x);
      minY=Math.min(minY,projected.y);maxY=Math.max(maxY,projected.y);
    }
    const spanX=maxX-minX,spanY=maxY-minY;
    const offset=new THREE.Vector3((minX+maxX)/2,(minY+maxY)/2,0).applyQuaternion(this.camera.quaternion);
    this.camera.position.add(offset);
    this.controls=this.createControls();
    this.controls.target.copy(offset);
    const aspect=Math.max(.2,this.host.clientWidth/Math.max(1,this.host.clientHeight));
    this.viewHeight=Math.max(spanY,spanX/aspect)*1.10;
    this.resize();
    this.controls.update();
    this.dirty=true;
  }
  setMode(mode:DisplayMode){this.mode=mode;this.uniforms.displayMode.value=mode==="Activity"?1:0;this.renderer.setClearColor(mode!=="Anatomy"?0x08121e:0xffffff);this.dirty=true}
  setRotation(value:boolean){this.rotation=value;this.controls.autoRotate=value;this.dirty=true}
  setActive(value:boolean){this.active=value;this.dirty=true}
  zoom(factor:number){this.camera.zoom=THREE.MathUtils.clamp(this.camera.zoom*factor,.6,12);this.camera.updateProjectionMatrix();this.dirty=true}
  updateActivity(view:BrainView|undefined,context:SampleContext):ActivityStats{
    if(this.neuralStream?.metadata)return this.signals.stats;
    const now=performance.now();
    // The legacy view uses measured changes too; never add random pulses.
    this.uniforms.highlightEnabled.value=context.running&&context.connected&&!this.reducedMotion?1:0;
    if(this.signals.accept(view,context,now)){
      this.smoothingUntil=now+350;
      this.settling=true;
      this.changesUntil=now+CHANGE_HIGHLIGHT_MS+40;
      this.texture.needsUpdate=true;
      if(this.mode==="Activity")this.dirty=true;
    }
    return this.signals.stats;
  }
  setNeuralStream(stream:NeuralStream|null){
    if(this.neuralStream!==stream){
      this.signals.pixels.fill(0);this.neuralMetadata=null;
      this.signals.stats={step:null,count:0,highlighted:0,maxChange:0,status:'offline'};
    }
    this.neuralStream=stream;this.texture.needsUpdate=true;this.dirty=true;
  }
  getActivityStats(){return this.signals.stats}
  private configureActivity(metadata:NeuralMetadata){
    const owners=new Map(this.data.manifest.neurons.map((n,i)=>[n.id,i]));
    this.neuralOwners=metadata.nodes.map(n=>owners.get(n.id)??-1);
    this.neuralMetadata=metadata;
    this.host.dataset.geometry='original-traced-swc';
    this.host.dataset.connections='0';
  }
  private updateNeuralFrame(time:number){
    const stream=this.neuralStream;if(!stream?.metadata)return;
    if(this.neuralMetadata!==stream.metadata)this.configureActivity(stream.metadata);
    const frame=stream.playback.frame(time);if(!frame)return;
    const {a,b,fraction}=frame;
    const history=stream.playback.samples;
    const previous=history[Math.max(0,history.indexOf(a)-1)]??a;
    const highlighting=frame.live&&!stream.error&&!this.reducedMotion&&a.step!==b.step;
    let highlighted=0,maxChange=0;
    for(let i=0;i<this.neuralOwners.length;i++){
      const owner=this.neuralOwners[i];if(owner<0)continue;
      const offset=owner*4;
      this.signals.pixels[offset]=activityLevel(a.activity[i]+(b.activity[i]-a.activity[i])*fraction);
      // Brighten the ORIGINAL curved arbor only in proportion to measured
      // rate changes, smoothly blended at the same buffered sample time.
      const change=highlighting?(1-fraction)*measuredActivityChange(previous.activity[i],a.activity[i])
        +fraction*measuredActivityChange(a.activity[i],b.activity[i]):0;
      this.signals.pixels[offset+1]=change;
      if(change>.05)highlighted++;
      maxChange=Math.max(maxChange,Math.abs(b.activity[i]-a.activity[i]));
      this.signals.pixels[offset+2]=0;this.signals.pixels[offset+3]=1;
    }
    this.uniforms.highlightEnabled.value=highlighting?1:0;this.settling=false;
    this.signals.stats={step:frame.step,count:this.neuralOwners.filter(i=>i>=0).length,highlighted,maxChange,status:stream.error||!frame.live?(stream.playback.running?'offline':'paused'):'live'};
    this.texture.needsUpdate=true;if(this.mode!=='Anatomy')this.dirty=true;
    this.host.dataset.neuralVersion='curved-live-activity-v2';this.host.dataset.changingNeurons=String(highlighted);this.host.dataset.neuralStep=String(frame.step);
    this.host.dataset.neuralPackets=String(stream.playback.count);this.host.dataset.neuralFeedAgeMs=frame.ageMs.toFixed(0);
    this.host.dataset.neuralFrames=String(++this.renderedFrames);this.host.dataset.neuralStatus=this.signals.stats.status;
  }
  private animate=(time:number)=>{
    if(this.disposed)return;
    this.frame=requestAnimationFrame(this.animate);
    if(!this.active||!this.visible||document.hidden||time-this.lastFrame<1000/30)return;
    const delta=Math.min(.1,(time-this.lastFrame)/1000);
    this.lastFrame=time;
    this.controls.update(delta);
    this.updateNeuralFrame(time);
    if(this.settling){
      const finished=this.reducedMotion||time>=this.smoothingUntil;
      this.signals.advance(finished?1:1-Math.exp(-delta/0.08));
      this.settling=!finished;
      this.texture.needsUpdate=true;
      if(this.mode==="Activity")this.dirty=true;
    }
    if(this.mode==="Activity"&&(time<this.changesUntil||this.lastRenderedTime<this.changesUntil)&&!this.reducedMotion)this.dirty=true;
    if(!this.dirty&&!this.rotation)return;
    const scale=this.host.clientHeight/(this.viewHeight/this.camera.zoom);
    this.uniforms.pointScale.value=scale*this.renderer.getPixelRatio();
    this.uniforms.sampleTime.value=time/1000;
    this.onScale(scale*100);
    this.renderer.render(this.scene,this.camera);
    this.lastRenderedTime=time;
    this.dirty=false;
  };
  dispose(){
    this.disposed=true;
    cancelAnimationFrame(this.frame);
    this.observer.disconnect();this.visibilityObserver.disconnect();this.controls.dispose();
    this.renderer.domElement.removeEventListener("keydown",this.onKey);
    this.renderer.domElement.removeEventListener("webglcontextlost",this.onContextLost);
    for(const geometry of this.geometries)geometry.dispose();
    for(const material of this.materials)material.dispose();
    this.texture.dispose();this.renderer.dispose();this.renderer.domElement.remove();
  }
}

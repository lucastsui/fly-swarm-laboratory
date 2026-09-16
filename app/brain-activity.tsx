"use client";

import {useEffect,useMemo,useRef,useState} from "react";

import {Button} from "@/components/ui/button";
import {Dialog,DialogContent,DialogTitle,DialogDescription,DialogTrigger} from "@/components/ui/dialog";
import {loadMorphology,MorphologyRenderer,type Morphology,type Preset,type DisplayMode} from "./morphology-renderer";
import {activityLevel,type ActivityStats,type SampleContext} from "./activity-signals";
import DashboardDetails from "./dashboard-details";
import {NeuralStream} from "./neural-stream";

export type BrainView={nodes:{id:number;group:number;position:[number,number,number]}[];edges:[number,number,number][];activity:number[];bounds:number[][]};
type Props={view?:BrainView;running:boolean;step?:number;runId?:string;connected?:boolean;api?:string;checkpointHash?:string};
const format=new Intl.NumberFormat("en-US");

function Viewer({data,view,running,step,runId,connected=true,mode,onModeChange,active=true,expanded=false,stream}:{data:Morphology;mode:DisplayMode;onModeChange:(value:DisplayMode)=>void;active?:boolean;expanded?:boolean;stream:NeuralStream|null}&Props){
  const host=useRef<HTMLDivElement>(null),renderer=useRef<MorphologyRenderer|null>(null),scale=useRef<HTMLDivElement>(null);
  const latestView=useRef(view);
  latestView.current=view;
  const sampleContext=useRef<SampleContext>({step,runId,running,connected});
  sampleContext.current={step,runId,running,connected};
  const [stats,setStats]=useState<ActivityStats>({step:null,count:0,highlighted:0,maxChange:0,status:"offline"});
  const [preset,setPreset]=useState<Preset>("Oblique");
  const [rotating,setRotating]=useState(false);
  const [error,setError]=useState("");
  const settings=useRef({mode,preset,rotating,active});
  settings.current={mode,preset,rotating,active};
  useEffect(()=>{
    if(!host.current)return;
    try{
      const drawing=new MorphologyRenderer(host.current,data,pixels=>{if(scale.current)scale.current.style.width=`${pixels}px`});
      renderer.current=drawing;
      setStats({...drawing.updateActivity(latestView.current,sampleContext.current)});
      drawing.setMode(settings.current.mode);
      drawing.setPreset(settings.current.preset);
      drawing.setRotation(settings.current.rotating);
      drawing.setActive(settings.current.active);
      return()=>{drawing.dispose();renderer.current=null};
    }catch(error){setError(error instanceof Error?error.message:"3D graphics could not start.")}
  },[data]);
  useEffect(()=>{
    if(renderer.current)setStats({...renderer.current.updateActivity(view,{step,runId,running,connected})});
  },[view,step,runId,running,connected]);
  useEffect(()=>{
    renderer.current?.setNeuralStream(stream);
    const timer=setInterval(()=>{if(renderer.current)setStats({...renderer.current.getActivityStats()})},500);
    return()=>clearInterval(timer);
  },[data,stream]);
  useEffect(()=>renderer.current?.setActive(active),[active]);
  useEffect(()=>renderer.current?.setMode(mode),[mode]);
  const changeMode=(value:DisplayMode)=>onModeChange(value);
  const changePreset=(value:Preset)=>{setPreset(value);renderer.current?.setPreset(value)};
  const liveIds=new Set(view?.nodes.map(n=>n.id)||[]);
  const liveCount=data.manifest.neurons.filter(n=>liveIds.has(n.id)).length;
  return <div className={`morphology-viewer${expanded?" expanded":""}${mode!=="Anatomy"?" activity-mode":""}`}>
    <div className="morphology-toolbar">
      <div className="morphology-options" aria-label="Neuron coloring">{(["Anatomy","Activity"] as DisplayMode[]).map(value=><Button key={value} size="sm" variant="ghost" aria-pressed={mode===value} onClick={()=>changeMode(value)}>{value==="Activity"?"Live activity":"Anatomy colors"}</Button>)}</div>
      <div className="morphology-options" aria-label="Camera view">{(["Oblique","Front","Side"] as Preset[]).map(value=><Button key={value} size="sm" variant="ghost" aria-pressed={preset===value} onClick={()=>changePreset(value)}>{value}</Button>)}</div>
    </div>
    <div className="morphology-stage">
      <div className="morphology-host" ref={host} data-testid="neural-renderer"/>
      <div className="morphology-view-label">{mode==="Anatomy"?"MALE CNS · 3D RECONSTRUCTION":"LIVE ACTIVITY · ORIGINAL TRACED BRANCHES"}<span>{mode==="Anatomy"?"Distinct color per neuron":`${format.format(liveCount)} monitored cells · ${stats.status==="live"?"Live":stats.status==="paused"?"Paused":"Offline"}`}</span>{mode==="Activity"&&<><span className="activity-sample">Measured step {stats.step===null?"—":format.format(stats.step)}</span><span className="activity-changes">{stats.status==="live"?`${format.format(stats.highlighted)} cells changing · fly 1`:"Activity held"}</span></>}</div>
      <div className="morphology-navigation">
        <Button variant="ghost" size="icon-sm" aria-label="Zoom in on neuron wiring" onClick={()=>renderer.current?.zoom(1.3)}>[+]</Button>
        <Button variant="ghost" size="icon-sm" aria-label="Zoom out from neuron wiring" onClick={()=>renderer.current?.zoom(1/1.3)}>[-]</Button>
        <Button variant="ghost" size="icon-sm" aria-label="Reset anatomy camera" onClick={()=>changePreset(preset)}>[reset]</Button>
        <Button variant="ghost" size="icon-sm" aria-label="Auto rotate brain" aria-pressed={rotating} onClick={()=>{setRotating(!rotating);renderer.current?.setRotation(!rotating)}}>[rotate]</Button>
      </div>
      <div className="morphology-scale"><div ref={scale}/><span>100 µm</span></div>
      <div className="morphology-gesture">Drag to rotate · Scroll to zoom · Shift-drag to pan</div>
      {error&&<div className="morphology-error" role="alert">The 3D view could not start: {error}<br/>Reload the page to try again.</div>}
    </div>
    {mode==="Activity"?<div className="activity-legend"><div className="activity-level-key"><span>ACTIVITY MAGNITUDE · LOG SCALE</span><div className="activity-color-scale"/><div className="activity-scale-ticks">{[0,.001,.01,.1,1].map(value=><span key={value} style={{left:`${activityLevel(value)*100}%`}}>{value}</span>)}</div></div><div className="activity-change-key"><span>Gold brightening = live activity changes</span><small>{stream?.metadata?"Up to 20 samples/s · smoothly interpolated":"Legacy neuron samples · high-rate feed unavailable"}</small><small>Original curved branches · measured neuron activity, not electrical spikes.</small></div></div>:<div className="morphology-legend"><i className="morphology-rainbow"/><span>Real traced branches · color identifies each neuron</span><span>Original geometry · unmirrored</span></div>}
  </div>;
}

export default function BrainActivity({view,running,step,runId,connected=true,api,checkpointHash}:Props){
  const [data,setData]=useState<Morphology|null>(null);
  const [progress,setProgress]=useState("Loading real neuron skeletons…");
  const [error,setError]=useState("");
  const [attempt,setAttempt]=useState(0);
  const [open,setOpen]=useState(false);
  const [mode,setMode]=useState<DisplayMode>("Activity");
  const stream=useMemo(()=>api&&checkpointHash?new NeuralStream(api,checkpointHash):null,[api,checkpointHash]);
  useEffect(()=>{stream?.start();return()=>stream?.stop()},[stream]);
  useEffect(()=>{
    let active=true;
    loadMorphology(message=>{if(active)setProgress(message)}).then(value=>{if(active)setData(value)}).catch(reason=>{if(active)setError(reason instanceof Error?reason.message:"Could not load anatomy.")});
    return()=>{active=false};
  },[attempt]);
  return <section className="brain-visual" id="brain-anatomy" aria-label="Anatomically traced fly brain and ventral nerve cord">
    <div className="panel-head"><div><span className="eyebrow">ANATOMY EXPLORER</span><h2>Fly brain & nerve cord</h2></div><Dialog open={open} onOpenChange={setOpen}><DialogTrigger asChild><Button variant="outline" size="sm" disabled={!data}>[enlarge]</Button></DialogTrigger><DialogContent className="morphology-dialog terminal-dialog"><DialogTitle>MaleCNS · traced neuron anatomy</DialogTitle><DialogDescription>{data?`${format.format(data.manifest.neuronCount)} released neuron skeletons. Rotate and zoom to inspect the curved arbors and connecting nerve bundles.`:"Loading morphology"}</DialogDescription>{data&&<Viewer data={data} view={view} running={running} step={step} runId={runId} connected={connected} stream={stream} mode={mode} onModeChange={setMode} expanded/>}<p className="morphology-dialog-note">Centerline skeletons, not cell surface meshes or individual synapse sites. Soma dots mark annotated positions; their size is illustrative.</p></DialogContent></Dialog></div>
    {data?<Viewer data={data} view={view} running={running} step={step} runId={runId} connected={connected} stream={stream} mode={mode} onModeChange={setMode} active={!open}/>:<div className="morphology-loading" role="status">{error?<><p>{error}</p><Button variant="outline" onClick={()=>{setError("");setAttempt(value=>value+1)}}>Retry anatomy download</Button></>:<><p>{progress}</p><span>Preparing the anatomical view</span></>}</div>}
    <div className="brain-anatomy-caption"><span>{data?`${format.format(data.manifest.neuronCount)} traced neurons · anatomy sample`:"MaleCNS v1.0 morphology"}</span><DashboardDetails label="About this view" title="Data source & display accuracy" description="Real curved centerline traces; a representative sample of the full connectome."><p>{data?`${format.format(data.manifest.neuronCount)} neurons · ${format.format(data.manifest.segmentCount)} traced segments`:"MaleCNS v1.0 morphology"}</p><p>All 166,700 graph neurons remain in the model. Curvature and branching come from released 3D centerline traces. Individual synapse sites and cell surface meshes are not displayed.</p><p>Original SWC branches are simplified along each unbranched path with a verified maximum deviation of 0.2 µm. Roots, branch points and tips are retained. No anatomical warping is applied. Coordinates are converted uniformly from 8 nm voxels to micrometers. Soma dots show annotated locations with a uniform illustrative size. Colors distinguish neurons; activity colors interpolate the sampled model readings, not biological spike measurements.</p><p>The original curved arbors are unchanged. Their colors show each neuron’s measured rate; gold brightening follows its measured sample-to-sample changes, with numerical jitter filtered. All branches of one neuron share that neuron’s reading. No straight graph overlay, electrical spikes, per-synapse locations or signal propagation along branches are invented. Up to 20 measurements per wall-clock second are smoothly interpolated at 30 frames per second with roughly 160 ms buffering. No training or brain weights are changed by this view.</p><p>MaleCNS collaboration: FlyEM / HHMI Janelia, University of Cambridge, MRC LMB and Google Research. <a href="https://male-cns.janelia.org/download/" target="_blank" rel="noreferrer">Released skeleton data ↗</a> · <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noreferrer">CC BY 4.0</a></p></DashboardDetails></div>
  </section>;
}

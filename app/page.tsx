"use client";
import {useCallback,useEffect,useRef,useState} from "react";
import {Button} from "@/components/ui/button";
import BrainActivity,{type BrainView} from "./brain-activity";
import SmoothAvatar from "./smooth-avatar";
import LayoutBoxes from "./layout-boxes";
import {ResponsiveContainer,LineChart,Line,XAxis,YAxis,Tooltip,CartesianGrid} from "recharts";

type State={experiment:string;runId:string;running:boolean;steps:number;simSeconds:number;elapsedSimSeconds:number;tickMilliseconds:number;device:string;gpuMemoryGB:number;error?:string;
 phase?:{phase:string;validationStatus:string;validationSummary:string;publishedAt:string};phaseReloadError?:string;returns?:number;
 checkpointHash:string;checkpointLabel:string;trainingMethod:string;seed:number;totalProducts:number;totalTransfers:number;totalPickups:number;episodes:number;
 continuousSupply:boolean;automaticReset:boolean;simulationSpeed?:number;lastDelivery?:{product:number;seconds:number};
 playbackMode?:string;measuredSimulationSpeed?:number;
 sensoryInterface?:string;layout?:{kind:string;positions:number[][]};layoutChanges?:number;
 layoutEvidence?:{seconds:number;summary:Record<string,{cases:number;products:number;worldsWithProduct:number;worldsWithThree:number}>};
 swarm?:{flies:number;collisions:boolean;bumpEvents:number;solverHolds:number;activityFly:number};
 swarmEvidence?:{seconds:number;seed:number;summary:{swarm:{cases:number;products:number;atLeastThree:number;withLateProduct:number}}};
 avatars?:{id:number;deliveries:number;cargoStage:number;trajectory:number[][]}[];
 evidence:{cases:number;seconds:number;cleared:number;baselineCleared:number;products:number;baselineProducts:number;continuousCases:number;continuousProducts:number;baselineContinuousProducts:number;limitations:string[];curve:{seconds:number;baseline:number;selected:number}[]};
 brain:{neurons:number;edges:number;sensory:number;meanActivity:number;weightChange:number;initialWeightHash:string;plasticSynapses:number;changedSynapses:number};
 avatar:{x:number;y:number;heading:number;speed:number;turn:number;interact:boolean;cargo:boolean;cargoStage:number;contact:boolean;distance:number;collisions:number;deliveries:number;rates:Record<string,number>};
 plane:{width:number;height:number;sensingRange:number;landmarks:{x:number;y:number;radius:number;kind:string;color:string;stock:number;timer:number}[];obstacles:number[][];trajectory:number[][]};
 sensory:number[];history:{seconds:number;reward:number;products:number}[];events:string[];motorNeurons:Record<string,number[]>;brainView:BrainView};

const APIS={latest:"http://127.0.0.1:8770/api/plane",reference:"http://127.0.0.1:8769/api/plane"};
const n=(value=0,d=0)=>value.toLocaleString("en-US",{maximumFractionDigits:d,minimumFractionDigits:d});

export default function Home(){
 const [data,setData]=useState<State|null>(null),[error,setError]=useState("Connecting to local brain service…");
 const [busy,setBusy]=useState(false),[audit,setAudit]=useState(""),[details,setDetails]=useState(false),[notice,setNotice]=useState("");
 const [chart,setChart]=useState<"reward"|"products"|"verification">("products");
 const [selection,setSelection]=useState<"latest"|"reference">("latest");
 const API=APIS[selection],activeApi=useRef(API);activeApi.current=API;
 const selectModel=async(value:typeof selection)=>{
  if(value===selection)return;
  activeApi.current=APIS[value];setSelection(value);setData(null);setAudit("");setNotice("");setChart("products");setError("Connecting to selected model…");setBusy(true);
  try{
   const call=(api:string,action:string)=>fetch(api+"/control",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({action}),signal:AbortSignal.timeout(5000)});
   // Keep the other factory/checkpoint, but do not spend GPU on an unseen view.
   try{await call(APIS[value==="latest"?"reference":"latest"],"pause")}catch{/* Selected viewer remains usable if the other is unavailable. */}
   const response=await call(APIS[value],"start");if(!response.ok)throw new Error("Selected view could not start");
  }catch(e){setNotice((e as Error).message)}finally{setBusy(false)}
 };
 const refresh=useCallback(async()=>{
  const response=await fetch(API+"/state",{cache:"no-store",signal:AbortSignal.timeout(10000)});
  const value=await response.json() as State&{detail?:string};
  if(!response.ok||!value.avatar)throw new Error(value.detail||"Waiting for measured activity");
  if(value.experiment!==(API===APIS.latest?"experimental-layout-phase-v1":"verified-whole-line-service-v1"))throw new Error("Unexpected model service; refusing to mix checkpoints.");
  if(!value.continuousSupply||value.automaticReset)throw new Error("Waiting for the continuous-run brain service…");
  if(activeApi.current===API){setData(value);setError(value.error||"")}return value;
 },[API]);
 useEffect(()=>{let active=true;let timer:ReturnType<typeof setTimeout>;
  const poll=async()=>{try{await refresh()}catch(e){if(active&&activeApi.current===API)setError((e as Error).message+" — selected viewer unavailable; the reference model is available separately.")}if(active)timer=setTimeout(poll,1000)};
  void poll();return()=>{active=false;clearTimeout(timer)};
 },[refresh,API]);
 const command=async(action:string,extra:Record<string,unknown>={})=>{
  setBusy(true);
  try{
   const response=await fetch(API+"/control",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({action,...extra})});
   if(!response.ok)throw new Error((await response.json() as {detail:string}).detail);
   setNotice(action==="save"?"Checkpoint verified on disk; already saved, unchanged.":"");
   await refresh();
  }catch(e){setNotice((e as Error).message)}finally{setBusy(false)}
 };
 const verify=async()=>{
  setBusy(true);try{
   const response=await fetch(API+"/audit");if(!response.ok)throw new Error("Audit unavailable");
   const value=await response.json();setAudit(JSON.stringify(value,null,2));
  }catch(e){setAudit((e as Error).message)}finally{setBusy(false)}
 };
 const avatar=data?.avatar,evidence=data?.evidence;
 const live=!!data?.running&&!error;
 const experimental=!!data?.phase;
 const speedLabel=data?.playbackMode==="max"?"MAX (actual "+n(data.measuredSimulationSpeed,1)+"×)":(data?.simulationSpeed??1)+"×";
 const layoutMode=!!data?.sensoryInterface?.startsWith("local-color-cargo-v")||data?.sensoryInterface==="annotated-color-cargo-status-v2";
 const chartRows:{seconds:number;baseline?:number;selected?:number;products?:number;reward?:number}[]=chart==="verification"?(evidence?.curve||[]):(data?.history||[]);
 const lines=[
  "> brain: "+(data?data.checkpointLabel+" / "+data.checkpointHash.slice(0,12):"connecting"),
  "> mode: "+(live?"LIVE INFERENCE":error?"OFFLINE":"PAUSED")+" | shown weights frozen | training runs separately",
  ...(data?.phase?["> phase: "+data.phase.phase+" | EXPERIMENTAL | validation "+data.phase.validationStatus,"> tests: "+data.phase.validationSummary,"> each new phase loads automatically; new phase = fresh display factory"]:["> reference: prior original-layout model; not arbitrary-layout verification"]),
  "> swarm: "+(data?.swarm?.flies??1)+" flies | shared brain weights | separate neural states",
  "> collisions: fly ↔ fly | "+n(data?.swarm?.bumpEvents)+" bumps | stations collisionless",
  "> factory: "+speedLabel+" | "+n(data?.simSeconds,1)+"s | "+n(data?.totalProducts)+" products | no resets",
  "> live totals: "+n(data?.totalPickups)+" pickups | "+n(data?.totalTransfers)+" transfers | "+n(data?.totalProducts)+" products",
  "> fly 1: "+(avatar?.cargo?["","ore","plate","component"][avatar.cargoStage]:"empty")+" | speed "+n(avatar?.speed,3)+" | turn "+n(avatar?.turn,3),
  "> trained: supervised brain "+(experimental?"synapses + excitability":"synaptic gains")+"; not dopamine",
  "> brain view: measured activity of fly 1 (not swarm average)",
  ...(!experimental?["> test: "+(data?.layoutEvidence?Object.entries(data.layoutEvidence.summary).map(([k,v])=>k+" "+v.worldsWithProduct+"/"+v.cases).join(" | "):data?.swarmEvidence?data.swarmEvidence.summary.swarm.atLeastThree+"/"+data.swarmEvidence.summary.swarm.cases+" worlds made ≥3 products in "+data.swarmEvidence.seconds+"s":"verification pending")]:["> physical returns: "+n(data?.returns)+" | source trays | no automatic rescue"]),
  "> layout: "+(data?.layout?.kind??"original")+(layoutMode?" | drag boxes / arrow keys; ≥1.6 units apart":" | original-layout brain"),
  ...(notice?["> "+notice]:[]),
  ...(data?.phaseReloadError?["> latest phase failed to load; retaining shown checkpoint: "+data.phaseReloadError]:[]),
  ...(error?["> ERROR: "+error]:[])
 ];
 const method=experimental?[
  "LATEST TRAINING PHASE / EXPERIMENTAL — NOT A RELIABILITY CLAIM",
  data?.checkpointLabel,"Parameter SHA256: "+data?.checkpointHash,
  "Validation: "+data?.phase?.validationStatus,data?.phase?.validationSummary,
  "Published: "+data?.phase?.publishedAt,
  "",data?.trainingMethod,
  "297 fixed local sensory channels → full 166,700-neuron signed connectome → fixed motor decoder.",
  "No external decision network, target-selected goal, teacher, optimizer or exploration in the display.",
  "Four colliding flies share weights but have separate neural state; boxes are collisionless.",
  "Brain-controlled pickup, transfer and physical returns; no teleportation, disposal or automatic rescue.",
  "Local box colors, cargo, output, capacity and processing cues are visible within 6 units; no global map.",
  "Drag or shuffle boxes to test behavior. Cargo, stock, processing and brain states are retained.",
  "A new phase starts a fresh display factory and clears the displayed trace/chart. Deliveries never reset it.",
  "Validation-only metadata updates do not reset the factory. A failed load retains the last working model.",
  "Reference model remains available separately; its original-layout physics and senses differ.",
  "Switching models pauses the unseen display to free GPU compute, while retaining its factory state.",
  "Training and independent validation continue separately; pause view does not pause training.",
  "Anatomy color and motor readings are measured activity from fly 1, not an average or invented firing.",
  "Live reward is logged, not used to update this frozen display checkpoint.",
  "Playback: "+speedLabel+". MAX removes wall-clock pacing, not neural or physical steps. Original movement outputs and 0.05s physics are unchanged; display readback is capped at 10 Hz.",
  "Laptop GPU: "+data?.device+" | "+n(data?.tickMilliseconds,1)+" ms/tick",
  "", "CURRENT MOTOR ACTIVITY",Object.entries(avatar?.rates||{}).map(([k,v])=>k+": "+n(v,6)).join(" | "),
  "Forward = clip(80 × DNg100 mean, 0, 2); turn = clip(160 × (right − left), −2, 2).",
  "Interaction when MN9 mean > 0.025; distance, stock and cooldown are enforced by the world.",
  "", "RECENT DISPLAY EVENTS",...(data?.events||[]),...(audit?["", "SHOWN-BRAIN AUDIT",audit]:[])
 ].join("\n"):[
  "SHARED BRAIN / COLLIDING SWARM",
  `${data?.swarm?.flies??1} flies share weights, but each retains its own neural state and senses.`,
  "Each fly sees nearby flies through its existing visual channels; impacts reach its contact channel.",
  "Swept circular bodies (radius 0.22), equal-mass inelastic bumps, collisionless stations.",
  "Shared stock and machine timers; simultaneous interactions rotate priority and cannot duplicate material.",
  "No traffic controller, inter-agent messages, assigned routes or automatic pickup/drop.",
  "The anatomical activity view and motor readings show fly 1, not an average of all flies.",
  data?.swarmEvidence?JSON.stringify(data.swarmEvidence,null,2):"Swarm verification pending.",
  ...(layoutMode?["RANDOM-LAYOUT EVALUATION",JSON.stringify(data?.layoutEvidence,null,2),
    "Engineered local-color and cargo interface, same brain graph and motor decoder.",
    "Box colors are visible only within 6 units. No global map or target vector is given to the brain.",
    "Changes retain stock, processing timers, cargo and neural state. Boxes require 0.8 border clearance and 1.6 separation.",
    "Performance is measured over sampled layouts, not guaranteed for every arrangement."]:[]),
  "",
  "HISTORICAL SINGLE-FLY VERIFICATION (NOT SWARM RESULTS)",
  data?.checkpointLabel||"Waiting for checkpoint",
  "Gain SHA256: "+(data?.checkpointHash||"—"),
  evidence?`${evidence.cleared}/${evidence.cases} fresh factories cleared all 3 items in ${evidence.seconds} simulated seconds (previous best: ${evidence.baselineCleared}/${evidence.cases}).`:"Waiting for paired evidence.",
  evidence?`${evidence.products}/192 finite products; continuously replenished source: ${evidence.continuousProducts} vs ${evidence.baselineContinuousProducts} products across ${evidence.continuousCases} trials.`:"",
  "Paired frozen tests: same seeds, no teacher, exploration or weight updates.",
  ...(evidence?.limitations||[]),
  "",
  "HOW THIS BRAIN WAS TRAINED",
  data?.trainingMethod||"Sequence supervised synaptic backpropagation with old-task replay.",
  n(data?.brain.changedSynapses)+" synaptic gains changed from the prior joint checkpoint; "+n(data?.brain.plasticSynapses)+" eligible.",
  layoutMode?"Graph routes, signs, neuron dynamics and motor decoder stay fixed; sensory projection now includes local color and cargo type.":"Graph routes, signs, sensory projection, neuron dynamics and motor decoder stay fixed.",
  "Backpropagation trained synaptic strengths, NOT a separate actor or decoder.",
  "This checkpoint was not trained with dopamine-style plasticity.",
  "",
  "LIVE DISPLAY",
  "All avatars and fly 1 anatomy activity come from this exact frozen checkpoint on the laptop GPU.",
  (layoutMode?"Local color, cargo and material sensing":"Original 30 sensory channels")+" → 166,700 neurons → fixed motor decoder → 2D movement and interaction.",
  "No teacher, routing controller, automatic pickup/drop, optimizer or learning in this view.",
  "One continuous factory: raw ore is replenished to 3; finished products accumulate without a limit.",
  "Deliveries do not reset the avatar, factory, neural state or elapsed time. There is no time limit.",
  "Speed and turning are continuous-valued outputs. Original physics runs at 20 Hz.",
  "Playback speed: "+(data?.simulationSpeed??1)+"×. The brain and entire factory advance together; physical steps and learned motor outputs are unchanged.",
  "The display continuously interpolates recorded positions with an adaptive 240–750 ms jitter buffer; it does not predict paths.",
  "The verification chart is measured clearance over simulated time, not a training loss curve.",
  "Live product/reward charts accumulate within this same factory; reward is logged only.",
  "Pause view stops this display only. Current experimental training and monitoring are separate.",
  "This historical reference is not evidence of arbitrary-layout reliability.",
  "",
  "CURRENT MOTOR ACTIVITY",
  Object.entries(avatar?.rates||{}).map(([key,value])=>key+": "+n(value,6)).join(" | "),
  "Forward = clip(80 × DNg100 mean, 0, 2); turn = clip(160 × (right − left), −2, 2).",
  "Interaction when MN9 mean > 0.025; the world enforces distance, stock and cooldown.",
  data?.lastDelivery?`Latest delivery: product ${data.lastDelivery.product} at ${n(data.lastDelivery.seconds,1)}s; continuing in the same world.`:"No live product delivered yet.",
  "Laptop GPU: "+(data?.device||"—")+" | "+n(data?.tickMilliseconds,1)+" ms/tick.",
  "",
  "SENSORY CHANNELS (CURRENT DISPLAY)",
  (data?.sensory||[]).map((v,i)=>i+":"+n(v,3)).join("  "),
  "",
  "RECENT DISPLAY EVENTS",
  ...(data?.events||[]),
  ...(audit?["","SHOWN-BRAIN AUDIT",audit]:[])
 ].join("\n");
 return <main className="terminal-lab">
  <header className="terminal-header">
   <h1>swarm@lab:~/assembly-line</h1>
   <nav aria-label="Experiment controls">
    <Button variant="ghost" aria-pressed={selection==="latest"} disabled={busy} onClick={()=>selectModel("latest")}>[latest phase]</Button>
    <Button variant="ghost" aria-pressed={selection==="reference"} disabled={busy} onClick={()=>selectModel("reference")}>[reference model]</Button>
    <Button variant="ghost" disabled={!data||busy||!!error} onClick={()=>void command(data!.running?"pause":"start")}>[{data?.running?"pause view":"resume view"}]</Button>
    <span>{speedLabel} / {selection==="latest"?"experimental":"reference"}</span>
    <Button variant="ghost" disabled={!data||busy||!!error} onClick={()=>void command("save")}>[save shown brain]</Button>
   </nav>
  </header>
  <section className="terminal-visuals">
   <section className="terminal-plane"><h2>assembly-line / {data?.swarm?.flies??1} flies · {n(data?.totalProducts)} products · {n(data?.swarm?.bumpEvents)} bumps {layoutMode&&<><Button variant="ghost" disabled={busy} onClick={()=>void command('randomize',{kind:'wide'})}>[shuffle boxes]</Button><Button variant="ghost" disabled={busy} onClick={()=>void command('randomize',{kind:'original'})}>[line]</Button></>}</h2>
    <div className="observation-plane"><svg viewBox="0 0 800 560" role="img" aria-label="A swarm of colliding flies sharing one assembly line, with local sensory cues and measured trajectories."><defs><pattern id="plane-grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M40 0H0V40" fill="none" stroke="#26394a" strokeWidth=".7"/></pattern></defs><rect width="800" height="560" fill="url(#plane-grid)"/><rect x="8.8" y="8.8" width="782.4" height="542.4" rx="4" fill="none" stroke="#405365"/><text x="24" y="34" className="map-meta">LOCAL SENSING / RED BODY = CONTACT / BRAIN VIEW = FLY 1</text>
     {data?.plane.obstacles.map(([x,y,r],i)=><circle key={i} cx={x*40} cy={y*40} r={r*40} fill="#263b4a" stroke="#667d8d"/>)}
     {data&&<LayoutBoxes boxes={data.plane.landmarks} editable={layoutMode&&!busy} onChange={positions=>command('layout',{positions})}/>}
     {data&&<SmoothAvatar api={API} checkpointHash={data.checkpointHash} sensingRange={data.plane.sensingRange} flies={data.swarm?.flies??1} trailSeed={{sessionId:data.runId,step:data.steps,avatars:data.avatars??[{id:0,trajectory:data.plane.trajectory}]}}/>}
     {!data&&<text x="400" y="280" textAnchor="middle" className="map-meta">Waiting for measured neural output…</text>}
    </svg></div>
   </section>
   <BrainActivity view={data?.brainView} running={live} step={data?.steps} runId={data?.runId} connected={!!data&&!error} api={API} checkpointHash={data?.checkpointHash}/>
  </section>
  <section className="terminal-lower">
   <section className="terminal-plot" aria-label="Measured performance curves">
    <nav aria-label="Chart selection">{(["products","reward","verification"] as const).filter(value=>value!=="verification"||!!evidence).map(value=><Button variant="ghost" key={value} aria-pressed={chart===value} onClick={()=>setChart(value)}>[{value==="verification"?"past single-fly tests":value==="products"?"swarm products":"swarm reward"}]</Button>)}</nav>
    <div className="terminal-chart">
    {data?<ResponsiveContainer width="100%" height="100%"><LineChart data={chartRows} margin={{top:16,right:16,left:0,bottom:4}}>
     <CartesianGrid vertical={false} stroke="#26372e"/><XAxis dataKey="seconds" stroke="#94ad9f" minTickGap={40} tickFormatter={v=>v+"s"}/>
     <YAxis domain={chart==="verification"?[0,100]:chart==="products"?[0,"auto"]:["auto","auto"]} allowDecimals={chart!=="products"} stroke="#94ad9f" tickFormatter={v=>chart==="verification"?v+"%":n(v,chart==="reward"?1:0)}/>
     <Tooltip labelFormatter={v=>v+" simulated seconds"} contentStyle={{background:"#09110d",border:"1px solid #41634e"}}/>
     {chart==="verification"?<><Line dataKey="selected" name="Selected brain: all 3 cleared %" stroke="#98dda9" dot={true} isAnimationActive={false}/><Line dataKey="baseline" name="Previous best: all 3 cleared %" stroke="#d5b77e" dot={true} isAnimationActive={false}/></>:<Line type={chart==="products"?"stepAfter":"linear"} dataKey={chart} name={chart==="products"?"Cumulative live products":"Cumulative live reward (no learning)"} stroke="#98dda9" dot={false} isAnimationActive={false}/>}
    </LineChart></ResponsiveContainer>:<pre>Waiting for selected brain service…</pre>}
    </div>
    <p>{chart==="verification"?"Historical finite tests: all 3 items / 64 worlds. Green: selected. Amber: previous best.":chart==="products"?"Finished products in the same continuous factory; x = elapsed simulated time.":"Cumulative task reward logged during live inference; no weight updates."}</p>
   </section>
   <section className="terminal-console" aria-label="Training terminal">
    <nav aria-label="Terminal output"><span>stdout / {details?"experiment details":"live status"}</span><Button variant="ghost" aria-pressed={details} onClick={()=>setDetails(!details)}>[{details?"status":"details"}]</Button><Button variant="ghost" disabled={!data||busy} onClick={()=>{setDetails(true);void verify()}}>[audit shown brain]</Button></nav>
    <pre tabIndex={0} aria-label={details?"Experiment details":"Live training text"}>{details?method:lines.join("\n")}</pre>
   </section>
  </section>
 </main>;
}

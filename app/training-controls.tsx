"use client";
import {useState} from "react";
import {Pause,Play,RotateCcw} from "lucide-react";
import {Slider} from "@/components/ui/slider";
import {Switch} from "@/components/ui/switch";
import {AlertDialog,AlertDialogTrigger,AlertDialogContent,AlertDialogHeader,AlertDialogTitle,AlertDialogDescription,AlertDialogFooter,AlertDialogCancel,AlertDialogAction} from "@/components/ui/alert-dialog";
import DashboardDetails from "./dashboard-details";

type Settings={flyCount:number;environmentCount:number;speed:number;unlimited:boolean;learning:boolean;cluster?:{selectedWorker:string;workers:{id:string;label:string}[]}};
export default function TrainingControls({settings,available,busy,live,act}:{settings?:Settings;available:boolean;busy:boolean;live:boolean;act:(action:string,values?:Record<string,unknown>)=>void}){
  const [flies,setFlies]=useState(settings?.flyCount??4);
  const [factories,setFactories]=useState(settings?.environmentCount??8);
  const [speed,setSpeed]=useState(settings?.speed??50);
  const total=flies*factories,valid=total<=128;
  const worker=settings?.cluster?.workers.find(w=>w.id===settings.cluster?.selectedWorker)?.label;
  return <section className="control-panel">
    <h2>Experiment controls</h2>
    <p className="help control-scope">Applies to all training GPUs</p>
    <button className="run-button" disabled={!available||busy} onClick={()=>act(live?"pause":"start")}>{live?<Pause size={18}/>:<Play size={18}/>} {busy?"Applying…":live?"Pause training":"Start training"}</button>
    <div className="control-label"><label htmlFor="learning-toggle">Learning enabled</label><Switch id="learning-toggle" checked={settings?.learning??true} disabled={!available||busy} onCheckedChange={learning=>act("configure",{learning})}/></div>
    <div className="control-label"><label htmlFor="unlimited-toggle">Maximum throughput</label><Switch id="unlimited-toggle" checked={settings?.unlimited??true} disabled={!available||busy} onCheckedChange={unlimited=>act("configure",{unlimited})}/></div>
    {!settings?.unlimited&&settings&&<><div className="control-label"><label>Speed cap</label><strong>{speed} ticks/s</strong></div><Slider aria-label="Maximum factory ticks per second" min={1} max={200} value={[speed]} onValueChange={v=>setSpeed(v[0])} onValueCommit={v=>act("configure",{speed:v[0]})}/></>}
    <DashboardDetails label="Next-run settings" title="Configure the next cluster run" description={`Watching ${worker??"DGX Spark 1"}. Changes are applied only after you confirm a new run.`}><div className="next-run-settings"><span className="eyebrow">NEXT RUN · {worker??"DGX SPARK 1"}</span>
      <div className="control-label"><label>Parallel factories</label><strong>{factories}</strong></div>
      <Slider aria-label="Parallel factories for next run" min={1} max={32} step={1} value={[factories]} onValueChange={v=>setFactories(v[0])}/>
      <div className="control-label"><label>Flies per factory</label><strong>{flies}</strong></div>
      <Slider aria-label="Flies per factory for next run" min={1} max={32} step={1} value={[flies]} onValueChange={v=>setFlies(v[0])}/>
      <p className={`help${valid?"":" setting-error"}`} aria-live="polite">{factories} × {flies} = {total} independent fly states. {valid?"One shared set of weights.":"Reduce the total to 128 or fewer."}</p>
      <p className="help">Couriers cooperate inside each factory. Factories have separate materials, machines and rewards.</p>
      <AlertDialog><AlertDialogTrigger asChild><button className="secondary-button" disabled={!available||busy||!valid}><RotateCcw size={15}/> New cluster run</button></AlertDialogTrigger><AlertDialogContent><AlertDialogHeader><AlertDialogTitle>Set {worker??"DGX Spark 1"} to {factories} factories with {flies} flies each?</AlertDialogTitle><AlertDialogDescription>Archive the cluster checkpoint, then reset every worker’s factories and the shared reward chart. Only this machine’s factory count changes. Keep the learned brain weights and shared optimizer. All workers will be paused until you start training.</AlertDialogDescription></AlertDialogHeader><AlertDialogFooter><AlertDialogCancel>Cancel</AlertDialogCancel><AlertDialogAction onClick={()=>act("reset",{flyCount:flies,environmentCount:factories,speed,unlimited:settings?.unlimited??true})}>Create cluster run</AlertDialogAction></AlertDialogFooter></AlertDialogContent></AlertDialog>
    </div></DashboardDetails>
  </section>;
}

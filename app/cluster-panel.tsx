import {Cpu,Network} from "lucide-react";
import DashboardDetails from "./dashboard-details";

export type ClusterWorker={id:string;label:string;host:string;environmentCount:number;flyCount:number;agents:number;status:string;actionsPerSecond:number;acceptedUpdates:number;sampledTransitions:number;rejectedUpdates:number;modelVersion:number;lastGradientLag:number;gpu:string;memoryGB:number;error?:string;communicationMs:number};
export type ClusterState={selectedWorker:string;workers:ClusterWorker[];modelVersion:number;totalAgents:number;totalFactories:number;parameterHash:string;maxGradientLag:number;method:string;rewardWindowSeconds:number};
const number=(n:number)=>n.toLocaleString("en-US",{maximumFractionDigits:0});
export default function ClusterPanel({cluster}:{cluster?:ClusterState}){
  if(!cluster)return null;
  return <section className="cluster-panel" aria-label="Distributed training workers">
    <div className="cluster-heading"><div><Network size={18}/><h2>One learner · {cluster.workers.length} GPUs</h2></div><span>Shared model v{number(cluster.modelVersion)}</span><DashboardDetails label="Worker details" title="Distributed training workers" description="Independent factories contribute to one shared optimizer."><div className="worker-details">{cluster.workers.map(worker=><article key={worker.id}><h2>{worker.label} · {worker.status}</h2><p className="help">{worker.environmentCount} factories × {worker.flyCount} flies · {worker.gpu}</p><dl><div><dt>Accepted updates</dt><dd>{number(worker.acceptedUpdates)}</dd></div><div><dt>Gradient round trip</dt><dd>{number(worker.communicationMs)} ms</dd></div><div><dt>Last gradient lag</dt><dd>{worker.lastGradientLag} / {cluster.maxGradientLag}</dd></div><div><dt>Rejected updates</dt><dd>{number(worker.rejectedUpdates)}</dd></div></dl>{worker.error&&<p className="setting-error">{worker.error}</p>}</article>)}</div><p className="help">Rates are measured. Paused or disconnected workers contribute zero displayed throughput. Start, pause and learning controls apply to all workers.</p></DashboardDetails></div>
    <div className="cluster-workers">{cluster.workers.map(worker=><article key={worker.id} className={`cluster-worker ${worker.id===cluster.selectedWorker?"watched":""}`}>
      <div className="cluster-worker-title"><strong><Cpu size={16}/>{worker.label}</strong><span className={worker.status==="Training"?"mint":""}>{worker.error?"Needs attention":worker.status}</span></div>
      <div className="cluster-rate">{number(worker.actionsPerSecond)}<small>actions / s</small></div>
      <p>{worker.environmentCount} factories × {worker.flyCount} flies <span>· {number(worker.acceptedUpdates)} updates</span></p>
      {worker.error&&<p className="setting-error" role="status">{worker.error}</p>}
    </article>)}</div>
  </section>;
}

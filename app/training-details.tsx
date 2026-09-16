"use client";
import DashboardDetails from "./dashboard-details";

export type Training={running:boolean;focus:string;version:number;maxUpdates:number;plasticSynapses:number;changedSynapses:number;
 configuration?:{rule:string;stage:string;horizon:number;eta:number};
 workers:{id:string;accepted:number;lastSeen:number;actionsPerSecond:number;contributionL1:number;device:string}[];
 evaluations:{version:number;carry:{successRate:number;episodes:number};fullLine:{pickups:number;transfers:number;products:number;episodes:number;productSuccessRate?:number;transferSuccessRate?:number;meanReward?:number}}[]};

export default function TrainingDetails({training}:{training:Training}){
 return <DashboardDetails label="Shared training details" title="All-synapse dopamine training" description="One canonical brain receives experience updates from the laptop and both DGX Sparks.">
  <div className="observation-method">
   <h3>{training.running?"Training enabled":"Training paused"} · update {training.version} / {training.maxUpdates}</h3>
   <p>{training.plasticSynapses.toLocaleString()} independently adjustable synaptic strengths; {training.changedSynapses.toLocaleString()} currently differ from the original connectome. No region-specific plasticity mask remains. An eligible weight need not change on every step.</p>
   <h3>GPU contributions</h3>
   {training.workers.map(w=><p key={w.id}>{w.id}: {w.accepted} accepted updates · last contact {Math.max(0,Math.round(Date.now()/1000-w.lastSeen))} s ago · {Math.round(w.actionsPerSecond)} simulation steps/s during its last rollout · {w.device}</p>)}
   {!training.workers.length&&<p>Waiting for the first submitted experience batch.</p>}
   <h3>Learning rule and safeguards</h3>
   <p>Pre/post activity leaves an eligibility trace at every existing connection. Reward stimulates identified dopamine populations; their measured evoked activity gates the updates. Extending this modulation to the whole graph is an experimental engineering choice, not a biologically validated dopamine receptor map.</p>
   <p>Routes, excitatory/inhibitory signs, neural dynamics, sensory projection, tonic drive and motor decoder remain fixed. Each log-strength gain is bounded to −2…+2 relative to its original strength. The existing slow-modulator fast-current mask is unchanged. No backpropagation or learned external controller is used.</p>
   <p>Current curriculum: start carrying a raw item near the first receiving station. Real progress and successful delivery earn reward; motor noise supports exploration. These short carrying trials do not demonstrate a complete assembly line.</p>
   <h3>Noise-free candidate tests</h3>
   <p>The 2D avatar, anatomy activity and pickup chart show the preserved, previously verified brain—not the changing candidate. Its displayed weights are frozen. Candidate tests below also disable noise and learning.</p>
   {training.evaluations.map((e,i)=><p key={i}>Version {e.version}: carrying success {Math.round(e.carry.successRate*100)}% in {e.carry.episodes} trials. Full-line test: {e.fullLine.pickups} pickups, {e.fullLine.transfers} transfers, {e.fullLine.products} finished products across {e.fullLine.episodes} worlds.</p>)}
   <p>Checkpoints save automatically. This stage pauses at {training.maxUpdates} accepted updates for review; it never overwrites or automatically replaces the verified pickup brain. Pause training affects the shared learner, while Pause simulation affects only the displayed brain.</p>
  </div>
 </DashboardDetails>;
}

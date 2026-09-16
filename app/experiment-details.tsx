"use client";
import {Button} from "@/components/ui/button";
import DashboardDetails from "./dashboard-details";

export type Study={selectedRound:number;checkpointHash:string;cases:number;pickups:number;transfers:number;products:number;
 discrimination:{cases:number;nearPickupRate:number;farAttemptRate:number;nearMeanLatency:number};
 history:{round:number;pickup:number;falseAttempt:number}[]};
type Evaluation={kind:string;episodes:number;pickups:number;transfers:number;products:number};

export default function ExperimentDetails({study,evaluations,onAudit,audit,busy}:{study:Study;evaluations:Evaluation[];onAudit:()=>void;audit:string;busy:boolean}){
 return <DashboardDetails label="Experiment results & limits" title="Useful pickup learning, not full hauling" description="Frozen weights and no exploratory noise in every reported test."><div className="observation-method">
  <h3>Fresh assembly-line trials</h3><p>Identical starts for the original and learned brains: {study.cases} worlds, 40 simulated seconds each. These seeds were not used for training or checkpoint selection.</p>
  <table className="experiment-table"><thead><tr><th>Brain</th><th>Pickups</th><th>Transfers</th><th>Products</th></tr></thead><tbody>{evaluations.map(e=><tr key={e.kind}><th>{e.kind}</th><td>{e.pickups}</td><td>{e.transfers}</td><td>{e.products}</td></tr>)}</tbody></table>
  <p>The learned brain picked up material in every tested world, and supplied a later station in 11 of 64 worlds. It did not finish the complete assembly line. Its movement circuit was not trained in this successful experiment; these transfers do not establish learned navigation.</p>
  <h3>Does it use its senses?</h3><p>A separate {study.discrimination.cases}-trial test used fresh positions: {Math.round(study.discrimination.nearPickupRate*100)}% nearby pickup success and {Math.round(study.discrimination.farAttemptRate*100)}% far-away interaction attempts. Removing all sensory input eliminated pickup and transfer behavior.</p>
  <p>An independent repeat on Spark 1 also achieved 128/128 nearby pickups and 0/128 far-away attempts. The matched Spark 2 control, with dopamine-gated updates disabled, retained 0/128 pickups and changed no synapses, despite exploratory pickups during training.</p>
  <h3>What changed</h3><p>All 166,700 neurons and 25,582,938 connections still run. During this pickup curriculum, only 415 existing synapses feeding the MN9 interaction neurons were allowed to change. The remaining weights, neuron dynamics, 30-channel sensory projection, and DNg100 / DNa02 / MN9 decoder stayed fixed. No backpropagation, learned decoder, steering script, or automatic pickup was added.</p>
  <p>Local presynaptic × postsynaptic activity was gated by reward-evoked activity in PAM01 / PPL101 dopamine populations. Updates preserved wiring and excitatory/inhibitory signs. This engineered dopamine-gated Hebbian rule is not a validated model of how real flies learn.</p>
  <p>Training trials rewarded one successful nearby pickup (+1) and penalized a far-away interaction (−0.1). Neural exploration began after eight ordinary movement ticks, allowing sensory activity to arrive first. Nothing supplied a target action to the decoder. The laptop and both Sparks ran candidate experiments and controls; the displayed checkpoint was trained on the laptop.</p>
  <h3>Why the checkpoint is frozen</h3><p>Round {study.selectedRound} was preserved using validation performance, before testing on fresh seeds. The plotted later rounds show why unlimited training is unsafe: continued updates eventually produced far-away false attempts. The current live simulation has no exploration and no weight updates. Play and Pause control inference only.</p>
  <p>Pickup is the first learned subskill. Reliable navigation, multistage delivery and full products remain unproven. Reward totals alone were misleading in this setup, so the chart reports actual pickup success and false attempts.</p>
  <Button variant="outline" onClick={onAudit} disabled={busy}>Audit preserved synapses</Button>{audit&&<p role="status">{audit}</p>}
 </div></DashboardDetails>;
}

import fs from 'node:fs';
import {performance} from 'node:perf_hooks';

const output=process.argv[2];
if(!output||fs.existsSync(output))throw new Error('Provide a fresh report path');
const start=performance.now(),records=[];
while(performance.now()-start<20000){
 const sent=performance.now()-start;
 const response=await fetch('http://127.0.0.1:8770/api/plane/motion');
 const packet=await response.json(),received=performance.now()-start;
 records.push({sent,received,packet});
 await new Promise(resolve=>setTimeout(resolve,Math.max(10,100-(received-sent))));
}
fs.writeFileSync(output,JSON.stringify(records));
const quantile=(v,q)=>[...v].sort((a,b)=>a-b)[Math.floor((v.length-1)*q)];
const latency=records.map(r=>r.received-r.sent),gaps=records.slice(1).map((r,i)=>r.received-records[i].received);
console.log(JSON.stringify({packets:records.length,latencyMs:{median:quantile(latency,.5),p95:quantile(latency,.95),max:Math.max(...latency)},
 arrivalGapMs:{median:quantile(gaps,.5),p95:quantile(gaps,.95),max:Math.max(...gaps)},
 lastSampleAgeMs:records.map(r=>r.packet.serverTimeMs-r.packet.samples.at(-1).timeMs).reduce((a,b)=>Math.max(a,b),0)},null,2));

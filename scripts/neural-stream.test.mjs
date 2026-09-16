import {test} from 'node:test';
import assert from 'node:assert/strict';
import {decodeNeuralSample,NeuralPlayback,weightedConnectionSignal,measuredActivityChange} from '../app/neural-stream.ts';

test('binary measurements preserve signed float32 data and reject invalid payloads',()=>{
 const bytes=new ArrayBuffer(40),v=new DataView(bytes);
 v.setUint32(0,0x31425346,true);v.setFloat64(4,81,true);v.setFloat64(12,12345,true);v.setUint32(20,2,true);
 [.01,.2,-.04,.1].forEach((x,i)=>v.setFloat32(24+i*4,x,true));
 const s=decodeNeuralSample(bytes);assert.equal(s.step,81);assert.equal(s.timeMs,12345);
 assert.ok(Math.abs(s.source[0]+.04)<1e-8);
 assert.throws(()=>decodeNeuralSample(bytes.slice(0,39)));
 v.setFloat32(24,NaN,true);assert.throws(()=>decodeNeuralSample(bytes));
 assert.equal(weightedConnectionSignal(-.5,.04),-.02);
});

const sample=(step,time,value)=>({step,timeMs:time,activity:new Float32Array([value]),source:new Float32Array([value*2])});
test('curved-arbor brightening follows measured changes, not a periodic flash',()=>{
 assert.equal(measuredActivityChange(.002,.002),0);
 assert.equal(measuredActivityChange(.002,.0020001),0);
 assert.equal(measuredActivityChange(NaN,.2),0);
 assert.ok(measuredActivityChange(.002,.0022)>0);
 assert.ok(measuredActivityChange(.002,.003)>measuredActivityChange(.002,.0022));
 assert.equal(measuredActivityChange(.003,.002),measuredActivityChange(.002,.003));
});
test('continuous playback interpolates measured values, never adds pulses or extrapolates',()=>{
 const p=new NeuralPlayback();p.accept(sample(1,1000,.1),'a',true,0);p.frame(0);
 p.accept(sample(2,1100,.2),'a',true,100);
 const frame=p.frame(150);assert.ok(frame.fraction>=0&&frame.fraction<=1);
 const value=frame.a.activity[0]+(frame.b.activity[0]-frame.a.activity[0])*frame.fraction;
 assert.ok(value>=.1&&value<=.201);
 for(let i=200;i<2000;i+=30){const f=p.frame(i);assert.ok(f.fraction<=1);assert.ok(f.b.timeMs<=1100);}
 const frozen=p.frame(2100);assert.equal(frozen.live,false);assert.equal(p.frame(2200).fraction,frozen.fraction);
});
test('pause and session changes cannot fabricate activity or mix histories',()=>{
 const p=new NeuralPlayback();p.accept(sample(8,1000,.1),'a',true,0);p.frame(0);
 p.accept(sample(9,1100,.2),'a',false,100);const held=p.frame(110);
 assert.equal(p.frame(200).fraction,held.fraction);assert.equal(p.count,2);
 p.accept(sample(1,2000,.8),'b',true,300);const f=p.frame(300);
 assert.equal(f.a.step,1);assert.equal(f.b.step,1);assert.equal(p.samples.length,1);
 p.accept(sample(1,2000,.9),'b',true,310);assert.equal(p.samples.length,1);
});

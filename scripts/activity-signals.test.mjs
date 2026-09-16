import {test} from "node:test";
import assert from "node:assert/strict";
import {ActivitySignals,activityLevel} from "../app/activity-signals.ts";

const sample=(a,b=0)=>({nodes:[{id:11},{id:22}],activity:[a,b]});
const context=(step,extra={})=>({step,runId:"test-run",running:true,connected:true,...extra});

test("log scale resolves small measured activations and retains sign-symmetric magnitude",()=>{
  assert.equal(activityLevel(0),0);
  assert.equal(activityLevel(1),1);
  assert.equal(activityLevel(-.003),activityLevel(.003));
  assert.ok(activityLevel(.0023)>.25);
  const values=[0,.00001,.0001,.001,.01,.1,1].map(activityLevel);
  assert.ok(values.every((x,i)=>!i||x>values[i-1]));
});

test("only meaningful changes after the baseline highlight; IDs own their data",()=>{
  const s=new ActivitySignals([22,11,33],4);
  s.accept(sample(.002),context(10),1000);
  assert.equal(s.stats.highlighted,0);
  assert.equal(s.pixels[4+3],1);
  assert.equal(s.pixels[8+3],0);
  s.accept(sample(.0022),context(11),1750);
  assert.equal(s.stats.highlighted,1);
  assert.ok(s.pixels[4+1]>.5);
  assert.equal(s.pixels[4+2],1.75);
  assert.equal(s.pixels[1],0);
  s.advance(1);
  assert.ok(Math.abs(s.pixels[4]-activityLevel(.0022))<1e-6);
});

test("unchanged values, duplicates, jitter, and a first sample do not invent events",()=>{
  const s=new ActivitySignals([11,22],2);
  s.accept(sample(.002),context(10),1000);
  s.accept(sample(.0020001),context(11),1750);
  assert.equal(s.stats.highlighted,0);
  s.accept(sample(.003),context(12),2500);
  const eventTime=s.pixels[2];
  assert.equal(s.accept(sample(.5),context(12),3250),false);
  assert.equal(s.pixels[2],eventTime);
  s.accept(sample(.003),context(13),4000);
  assert.equal(s.stats.highlighted,0);
  assert.equal(s.pixels[1],0);
});

test("pause, disconnect, new runs, gaps and step rollback suppress false bursts",()=>{
  const s=new ActivitySignals([11,22],2);
  s.accept(sample(.002),context(10),1000);
  s.accept(sample(.003),context(11),1750);
  assert.equal(s.stats.highlighted,1);
  s.accept(sample(.003),context(11,{running:false}),1800);
  assert.equal(s.stats.highlighted,0);
  assert.equal(s.pixels[1],0);
  s.accept(undefined,context(11,{connected:false}),2000);
  assert.equal(s.stats.status,"offline");
  s.accept(sample(.8),context(15),2500);
  assert.equal(s.stats.highlighted,0);
  s.accept(sample(.1),context(16),6000);
  assert.equal(s.stats.highlighted,0);
  s.accept(sample(.9),context(1,{runId:"new-run"}),6750);
  assert.equal(s.stats.highlighted,0);
  s.accept(sample(.1),context(0,{runId:"new-run"}),7500);
  assert.equal(s.stats.highlighted,0);
});

test("signed changes count even when absolute magnitude stays the same",()=>{
  const s=new ActivitySignals([11,22],2);
  s.accept(sample(.01),context(1),1000);
  s.accept(sample(-.01),context(2),1750);
  assert.equal(s.stats.highlighted,1);
  assert.equal(s.stats.maxChange,.02);
  assert.equal(s.pixels[1],1);
});

test("repeated paused or offline snapshots do not schedule more animation",()=>{
  const s=new ActivitySignals([11,22],2);
  s.accept(sample(.002),context(10),1000);
  assert.equal(s.accept(sample(.002),context(10,{running:false}),1750),true);
  assert.equal(s.accept(sample(.002),context(10,{running:false}),2500),false);
  assert.equal(s.accept(undefined,context(10,{connected:false}),3250),true);
  assert.equal(s.accept(undefined,context(10,{connected:false}),4000),false);
});

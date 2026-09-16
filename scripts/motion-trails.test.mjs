import {test} from 'node:test';
import assert from 'node:assert/strict';
import {MotionPlayback} from '../app/motion-samples.ts';
import {MotionTrails} from '../app/motion-trails.ts';
const sample=(step,x)=>({step,timeMs:step*50,x,y:7,heading:0,cargoStage:0,contact:false,deliveries:0});
const packet=(samples,sessionId='one')=>({sessionId,checkpointHash:'test',serverTimeMs:500,running:true,samples});
const avatar=(x,id=0)=>({id,x,y:7,heading:0,cargoStage:0,contact:false,deliveries:0});

test('trail advances between polls and ends exactly at the rendered fly, without future points',()=>{
 const playback=new MotionPlayback(),trails=new MotionTrails();
 const p=packet([sample(6,6),sample(7,7),sample(8,8),sample(9,9),sample(10,10)]);
 playback.accept(p,0);trails.accept(p);
 const first=playback.frame(5),next=playback.frame(15);
 const firstPath=trails.path({...first,id:0},first.step),nextPath=trails.path({...next,id:0},next.step);
 assert.notEqual(firstPath,nextPath);
 assert.ok(nextPath.endsWith(`L${(next.x*40).toFixed(3)} 280.000`));
 assert.ok(!nextPath.includes('400.000'));
});
test('initial history is preserved once; slow status cannot jump the trail forward',()=>{
 const trails=new MotionTrails();trails.accept(packet([sample(9,9),sample(10,10)]));
 trails.seed({sessionId:'one',step:10,avatars:[{id:0,trajectory:Array.from({length:11},(_,i)=>[i,7])}]});
 const path=trails.path(avatar(9.5),9);
 assert.ok(path.startsWith('M0.000 280.000'));
 assert.ok(path.endsWith('L380.000 280.000'));
 trails.seed({sessionId:'one',step:20,avatars:[{id:0,trajectory:[[999,7]]}]});
 assert.equal(trails.path(avatar(9.5),9),path);
});
test('overlapping packets deduplicate, each fly keeps its own trail, and memory is bounded',()=>{
 const trails=new MotionTrails();
 const p=packet(Array.from({length:1000},(_,step)=>({...sample(step,step),avatars:[avatar(step),avatar(-step,1)]})));
 trails.accept(p);trails.accept(p);
 assert.equal(trails.points.get(0).length,632);
 assert.equal(trails.points.get(1).length,632);
 assert.ok(trails.path(avatar(-998.5,1),998).endsWith('L-39940.000 280.000'));
});
test('restarts clear previous trails; missing measurements produce breaks, not invented lines',()=>{
 const trails=new MotionTrails();trails.accept(packet([sample(1,1),sample(4,4)]));
 assert.equal(trails.path(avatar(4.5),4),'M40.000 280.000 M160.000 280.000 L180.000 280.000');
 trails.accept(packet([sample(0,8)],'two'));
 trails.seed({sessionId:'one',step:10,avatars:[{id:0,trajectory:[[999,7]]}]});
 assert.equal(trails.path(avatar(8),0),'M320.000 280.000 L320.000 280.000');
});

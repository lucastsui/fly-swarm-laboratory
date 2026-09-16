import {test} from 'node:test';
import assert from 'node:assert/strict';
import {interpolatePose,MotionPlayback} from '../app/motion-samples.ts';
const pose=(step,timeMs,x,heading=0,extra={})=>({step,timeMs,x,y:7,heading,cargoStage:0,contact:false,deliveries:0,...extra});
const packet=(samples,extra={})=>({sessionId:'one',checkpointHash:'test',serverTimeMs:1000,running:true,samples,...extra});

test('interpolates continuous measured positions; never extrapolates',()=>{
 const samples=[pose(1,900,1),pose(2,950,1.1)];
 assert.equal(interpolatePose(samples,925).x,1.05);
 assert.equal(interpolatePose(samples,2000).x,1.1);
 assert.equal(interpolatePose(samples,0).x,1);
});
test('heading wraps by the shortest rotation and cargo switches only at its actual event',()=>{
 const samples=[pose(1,900,1,Math.PI-.1),pose(2,950,1.1,-Math.PI+.1,{cargoStage:1})];
 assert.ok(Math.abs(interpolatePose(samples,925).heading-Math.PI)<1e-10);
 assert.equal(interpolatePose(samples,949).cargoStage,0);
 assert.equal(interpolatePose(samples,950).cargoStage,1);
});
test('pause and stale connection hold the displayed pose',()=>{
 const p=new MotionPlayback();p.accept(packet([pose(1,800,1),pose(2,1000,2)]),100);
 const shown=p.frame(200);
 assert.equal(p.frame(300,false),shown);
 assert.equal(p.frame(1800),shown);
 p.accept(packet([pose(1,800,1),pose(2,1000,2)],{running:false}),1900);
 assert.equal(p.frame(1950),shown);
});
test('restart clears prior interpolation, packets cannot rewind motion',()=>{
 const p=new MotionPlayback();p.accept(packet([pose(10,900,10),pose(11,1000,11)]),0);p.frame(0);
 p.accept(packet([pose(9,850,9)]),1);assert.equal(p.packet.samples.at(-1).step,11);
 p.accept(packet([pose(0,1000,4)],{sessionId:'two'}),2);assert.equal(p.frame(2).x,4);
});
test('network timing jitter cannot move playback backwards',()=>{
 const p=new MotionPlayback(),s=[pose(1,700,1),pose(2,1000,2)];
 p.accept(packet(s),0);const first=p.frame(80).x;
 p.accept(packet(s,{serverTimeMs:1010}),120);assert.ok(p.frame(120).x>=first);
});
test('swarm interpolation keeps identities, cargo and contact independent',()=>{
 const a={id:0,x:1,y:7,heading:0,cargoStage:0,contact:false,deliveries:0};
 const b={...a,id:1,x:3,cargoStage:2};
 const samples=[pose(1,900,1,0,{avatars:[a,b]}),
  pose(2,1000,2,0,{avatars:[{...b,x:2,contact:true},{...a,x:2,cargoStage:1}]})];
 const middle=interpolatePose(samples,950).avatars;
 assert.deepEqual(middle.map(a=>a.id),[0,1]);
 assert.deepEqual(middle.map(a=>a.x),[1.5,2.5]);
 assert.deepEqual(middle.map(a=>a.cargoStage),[0,2]);
 assert.equal(middle[1].contact,false);
 assert.equal(interpolatePose(samples,1000).avatars[0].contact,true);
});

test('jittery arrivals do not reset the render clock or create stop/catch-up bursts',()=>{
 const playback=new MotionPlayback();
 const schedule=[0,50,100,150,260,310,360,410,510,560,610,690,740,790,850,900];
 let next=0,previous,smallest=Infinity,largest=0;
 for(let now=0;now<=1000;now+=10){
  while(next<schedule.length&&schedule[next]<=now){
   const received=schedule[next++],serverTime=received+1000-(next%3)*8;
   const end=Math.floor(serverTime/10);
   const samples=Array.from({length:70},(_,i)=>pose(end-69+i,(end-69+i)*10,(end-69+i)/100));
   playback.accept(packet(samples,{serverTimeMs:serverTime}),received);
  }
  const shown=playback.frame(now);
  if(previous!==undefined){const delta=shown.x-previous;smallest=Math.min(smallest,delta);largest=Math.max(largest,delta);}
  previous=shown.x;
 }
 assert.ok(smallest>0,'No artificial holds in an uninterrupted measured trajectory');
 assert.ok(largest/smallest<1.3,'No packet-arrival catch-up jumps');
 assert.equal(playback.underruns,0);
});

test('incremental packets merge bounded history and never extrapolate on a dropped connection',()=>{
 const playback=new MotionPlayback();
 playback.accept(packet(Array.from({length:30},(_,i)=>pose(i,i*10,i)),{serverTimeMs:290}),0);
 playback.frame(0);
 playback.accept(packet([pose(29,290,29),pose(30,300,30)],{serverTimeMs:300}),10);
 assert.equal(playback.packet.samples.length,31);
 for(let now=10;now<2000;now+=10){
  const shown=playback.frame(now);
  assert.ok(shown.x<=30);
 }
 const held=playback.frame(2200);assert.equal(playback.frame(2300),held);
});

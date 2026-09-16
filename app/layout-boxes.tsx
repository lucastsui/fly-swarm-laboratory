"use client";
import {useRef,useState} from "react";
type Box={x:number;y:number;radius:number;kind:string;color:string;stock:number;timer:number};
export default function LayoutBoxes({boxes,editable,onChange}:{boxes:Box[];editable:boolean;onChange:(positions:number[][])=>Promise<void>}){
 const [draft,setDraft]=useState<number[][]|null>(null);
 const drag=useRef<{index:number;points:number[][]}|null>(null);
 const begin=(event:React.PointerEvent<SVGGElement>,index:number)=>{
  if(!editable)return;event.preventDefault();event.currentTarget.setPointerCapture(event.pointerId);
  const points=boxes.map(b=>[b.x,b.y]);drag.current={index,points};setDraft(points);
 };
 const move=(event:React.PointerEvent<SVGGElement>)=>{
  if(!drag.current)return;
  const svg=event.currentTarget.ownerSVGElement,transform=svg?.getScreenCTM();if(!svg||!transform)return;
  const point=new DOMPoint(event.clientX,event.clientY).matrixTransform(transform.inverse());
  const points=drag.current.points.map(p=>[...p]);
  points[drag.current.index]=[Math.max(.8,Math.min(19.2,point.x/40)),Math.max(.8,Math.min(13.2,point.y/40))];
  drag.current.points=points;setDraft(points);
 };
 const end=()=>{
  const current=drag.current;drag.current=null;if(!current)return;
  void onChange(current.points).finally(()=>setDraft(null));
 };
 return <>{boxes.map((box,i)=>{
  const [x,y]=draft?.[i]??[box.x,box.y];
  return <g key={box.kind} role={editable?"button":undefined} tabIndex={editable?0:undefined}
   aria-label={box.kind+(editable?"; drag to move, or use arrow keys":"")}
   style={{cursor:editable?"grab":"default",touchAction:"none"}}
   onPointerDown={event=>begin(event,i)} onPointerMove={move} onPointerUp={end}
   onPointerCancel={()=>{drag.current=null;setDraft(null)}}
   onKeyDown={event=>{
    if(!editable||!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key))return;
    event.preventDefault();const points=boxes.map(b=>[b.x,b.y]);
    const axis=event.key==='ArrowLeft'||event.key==='ArrowRight'?0:1;
    points[i][axis]+=(event.key==='ArrowLeft'||event.key==='ArrowUp'?-.25:.25);
    void onChange(points);
   }}>
   <circle cx={x*40} cy={y*40} r={box.radius*40} fill={`${box.color}25`} stroke={box.color} strokeWidth="2"/>
   <text x={x*40} y={y*40+(i%2?-box.radius*40-16:box.radius*40+22)} textAnchor="middle" fill={box.color} className="plane-label">{box.kind}</text>
  </g>;
 })}</>;
}

"use client";

import type {ReactNode} from "react";
import {Dialog,DialogContent,DialogTitle,DialogDescription,DialogTrigger} from "@/components/ui/dialog";

export default function DashboardDetails({label,title,description,children}:{label:string;title:string;description:string;children:ReactNode}){
  return <Dialog><DialogTrigger asChild><button className="details-button">{label}</button></DialogTrigger><DialogContent className="dashboard-dialog"><DialogTitle>{title}</DialogTitle><DialogDescription>{description}</DialogDescription><div className="dashboard-dialog-body">{children}</div></DialogContent></Dialog>;
}

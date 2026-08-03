#!/usr/bin/env python3
"""Dependency-free local web window for Isaac crowd-map state."""

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
import webbrowser


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Isaac Sim 行人与无人机实时位置</title>
<style>
html,body{margin:0;height:100%;overflow:hidden;background:#18202a;color:#e8edf2;font:14px sans-serif}
#bar{height:42px;box-sizing:border-box;padding:11px 16px;background:#111820;display:flex;justify-content:space-between}
#map{display:block;width:100%;height:calc(100% - 42px)} .muted{color:#98aabe}
</style></head><body>
<div id="bar"><span id="status">正在等待仿真状态……</span>
<span class="muted">正常按先到先走；出现等待环时，环中组号最小者优先　◇ = 无人机</span></div><canvas id="map"></canvas>
<script>
const palette=['#2E86DE','#E67E22','#27AE60','#8E44AD','#C0392B','#16A085','#D4AC0D','#5D6D7E','#AF7AC5','#EC7063'];
const canvas=document.getElementById('map'),ctx=canvas.getContext('2d'),statusEl=document.getElementById('status');
let state=null;
function resize(){const d=devicePixelRatio||1,r=canvas.getBoundingClientRect();canvas.width=r.width*d;canvas.height=r.height*d;ctx.setTransform(d,0,0,d,0,0);draw();}
addEventListener('resize',resize);resize();
function color(g){return palette[Math.abs(Number(g)||0)%palette.length]}
function draw(){if(!state)return;const w=canvas.clientWidth,h=canvas.clientHeight,m=48;
 let pts=[...(state.walk_polygon||[]).map(p=>p.slice(0,2)),...(state.people||[]).map(p=>p.position.slice(0,2))];
 if(state.drone)pts.push(state.drone.position.slice(0,2));if(!pts.length)return;
 let xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]),xmin=Math.min(...xs)-1,xmax=Math.max(...xs)+1,ymin=Math.min(...ys)-1,ymax=Math.max(...ys)+1;
 let sc=Math.min((w-2*m)/Math.max(xmax-xmin,1),(h-2*m)/Math.max(ymax-ymin,1));
 const P=(x,y)=>[m+(x-xmin)*sc,h-m-(y-ymin)*sc];
 ctx.clearRect(0,0,w,h);ctx.fillStyle='#202a36';ctx.fillRect(0,0,w,h);ctx.font='11px sans-serif';ctx.textAlign='center';
 ctx.strokeStyle='#2a3745';ctx.lineWidth=1;ctx.fillStyle='#8fa3b8';
 for(let x=Math.floor(xmin);x<=Math.ceil(xmax);x++){let q=P(x,ymin);ctx.beginPath();ctx.moveTo(q[0],m);ctx.lineTo(q[0],h-m);ctx.stroke();ctx.fillText(x,q[0],h-20)}
 ctx.textAlign='right';for(let y=Math.floor(ymin);y<=Math.ceil(ymax);y++){let q=P(xmin,y);ctx.beginPath();ctx.moveTo(m,q[1]);ctx.lineTo(w-m,q[1]);ctx.stroke();ctx.fillText(y,31,q[1]+4)}
 let poly=state.walk_polygon||[];if(poly.length>2){ctx.beginPath();poly.forEach((p,i)=>{let q=P(p[0],p[1]);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.closePath();ctx.fillStyle='#263746';ctx.fill();ctx.strokeStyle='#6fa8c9';ctx.lineWidth=2;ctx.stroke()}
 for(const a of state.obstacles||[]){if(a.length<4)continue;let p=P(a[0],a[1]),q=P(a[2],a[3]);ctx.fillStyle='#56616d';ctx.fillRect(p[0],q[1],q[0]-p[0],p[1]-q[1]);ctx.strokeStyle='#7d8995';ctx.strokeRect(p[0],q[1],q[0]-p[0],p[1]-q[1])}
 for(const r of state.planned_routes||[]){let ps=r.points||[];if(ps.length<2)continue;ctx.beginPath();ps.forEach((p,i)=>{let q=P(p[0],p[1]);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.strokeStyle=color(r.group_id);ctx.globalAlpha=.42;ctx.lineWidth=2;ctx.setLineDash([8,5]);ctx.stroke();ctx.setLineDash([]);ctx.globalAlpha=1;
   for(const gate of r.gates||[]){let dx=ps[ps.length-1][0]-ps[0][0],dy=ps[ps.length-1][1]-ps[0][1],len=Math.hypot(dx,dy)||1,ux=dx/len,uy=dy/len,rad=gate[1],cx,cy;if(gate.length>=4&&gate[2]!==null&&gate[3]!==null){cx=gate[2];cy=gate[3]}else{let c=gate[0],mid=ps[Math.floor(ps.length/2)];cx=mid[0]+ux*(c-mid[0]*ux-mid[1]*uy);cy=mid[1]+uy*(c-mid[0]*ux-mid[1]*uy)}let a=P(cx-ux*rad,cy-uy*rad),b=P(cx+ux*rad,cy+uy*rad);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.strokeStyle=color(r.group_id);ctx.globalAlpha=.22;ctx.lineWidth=5;ctx.stroke();ctx.globalAlpha=1}
 }
 let groups=new Map();for(const p of state.people||[]){let k=String(p.group_id);if(!groups.has(k))groups.set(k,[]);groups.get(k).push(p)}
 for(const [g,ms] of groups){if(ms.length<2)continue;ms.sort((a,b)=>a.member_index-b.member_index);ctx.beginPath();ms.forEach((p,i)=>{let q=P(...p.position);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.strokeStyle=color(g);ctx.globalAlpha=.55;ctx.lineWidth=6;ctx.stroke();ctx.globalAlpha=1}
 for(const p of state.people||[]){let ms=groups.get(String(p.group_id)),grouped=ms.length>1,q=P(...p.position),r=grouped?9:7;if(p.traffic_waiting||p.planned_pause||p.traffic_cycle_override){ctx.beginPath();ctx.arc(q[0],q[1],r+5,0,Math.PI*2);ctx.strokeStyle=p.traffic_cycle_override?'#00e5ff':(p.traffic_waiting?'#ffd54f':'#ff9f43');ctx.lineWidth=3;ctx.stroke()}ctx.beginPath();ctx.arc(q[0],q[1],r,0,Math.PI*2);ctx.fillStyle=grouped?color(p.group_id):'#e8edf2';ctx.fill();ctx.strokeStyle='#111820';ctx.lineWidth=2;ctx.stroke();ctx.textAlign='left';ctx.font=(grouped?'bold ':'')+'12px sans-serif';ctx.fillStyle='#fff';ctx.fillText(p.name+(grouped?'  G'+p.group_id:'')+(p.traffic_cycle_override?'  环路优先放行':(p.traffic_waiting?'  等待通过':(p.planned_pause?'  计划停留':''))),q[0]+11,q[1]-11)}
 if(state.drone){let [x,y,z]=state.drone.position,q=P(x,y),s=12;ctx.beginPath();ctx.moveTo(q[0],q[1]-s);ctx.lineTo(q[0]+s,q[1]);ctx.lineTo(q[0],q[1]+s);ctx.lineTo(q[0]-s,q[1]);ctx.closePath();ctx.fillStyle='#00e5ff';ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke();ctx.fillStyle='#00e5ff';ctx.font='bold 13px sans-serif';ctx.textAlign='left';ctx.fillText(`OmniNxt  z=${z.toFixed(2)}m`,q[0]+15,q[1]+17)}
 let gc=[...groups.values()].filter(v=>v.length>1).length,waiting=(state.people||[]).filter(p=>p.traffic_waiting).length,pausing=(state.people||[]).filter(p=>p.planned_pause).length,age=Math.max(0,Date.now()/1000-state.wall_time);statusEl.textContent=`种子 ${state.seed} | 仿真 ${Number(state.simulation_time).toFixed(1)}s | 行人 ${state.people.length} | 多人群组 ${gc} | 交叉等待 ${waiting} | 计划停留 ${pausing} | 障碍 ${state.obstacles.length} | 状态延迟 ${age.toFixed(2)}s`;
}
async function poll(){try{let r=await fetch('/state',{cache:'no-store'});if(r.ok){state=await r.json();draw()}}catch(e){statusEl.textContent='与仿真状态服务断开，正在重试……'}setTimeout(poll,100)}poll();
</script></body></html>'''


class StateHandler(BaseHTTPRequestHandler):
    state_path = None

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            content = HTML.encode("utf-8")
            content_type = "text/html; charset=utf-8"
        elif self.path.startswith("/state"):
            try:
                content = Path(self.state_path).read_bytes()
            except OSError:
                content = b'{"ready":false,"people":[]}'
            content_type = "application/json; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, _format, *_args):
        return


def _watch_simulation(server, sim_pid):
    if not sim_pid:
        return
    while True:
        try:
            os.kill(sim_pid, 0)
        except OSError:
            server.shutdown()
            return
        time.sleep(1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--sim-pid", type=int)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    StateHandler.state_path = args.state
    server = ThreadingHTTPServer(("127.0.0.1", args.port), StateHandler)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Crowd map: {url}", flush=True)
    if args.sim_pid:
        threading.Thread(target=_watch_simulation, args=(server, args.sim_pid), daemon=True).start()
    if not args.no_browser:
        threading.Timer(0.2, lambda: webbrowser.open_new(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

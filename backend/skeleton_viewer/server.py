"""Dependency-light browser viewer for Alienware skeleton receiver output."""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_DIR = REPO_ROOT / ".local" / "run" / "skeleton_receiver"
COCO17_BONES = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OmniNxt Alienware 三维骨架</title>
<style>
html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#070b12;color:#eef4ff;font-family:system-ui,sans-serif}
#view{display:block;width:100%;height:100%;cursor:grab}#view.drag{cursor:grabbing}
.hud{position:fixed;left:16px;top:14px;padding:11px 14px;border:1px solid #314158;border-radius:10px;background:#0d1521dd;line-height:1.45;pointer-events:none;backdrop-filter:blur(5px)}
.title{font-weight:700;font-size:17px}.status{font-size:13px;color:#b8c8dc}.axes{margin-top:5px;font-size:12px}.x{color:#ff6c6c}.y{color:#67e886}.z{color:#65bfff}.bad{color:#ff936d}.good{color:#76f3a0}.hint{position:fixed;right:14px;bottom:12px;color:#9baabd;font-size:12px;background:#0d1521bb;padding:7px 10px;border-radius:7px;pointer-events:none}
</style></head><body>
<canvas id="view"></canvas>
<div class="hud"><div class="title">OmniNxt 12点身体骨架（不含头部）</div><div id="status" class="status">等待接收数据…</div><div class="axes"><span class="x">+X FRONT</span> · <span class="y">+Y LEFT</span> · <span class="z">+Z UP</span><br>XY 为地面平面 · base_link · metre<br>因果时序滤波 + 骨长约束 + 站立行走先验</div></div>
<div class="hint">左键拖动旋转 · 右键拖动平移 · 滚轮缩放 · R 重置</div>
<script>
const canvas=document.querySelector('#view'),ctx=canvas.getContext('2d'),statusEl=document.querySelector('#status');
const bones=[[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
const palette=['#ff7373','#65e88a','#67bfff','#e47ae8','#ffd166','#57e5de','#ff9f68','#a8e063'];
let frame={state:'waiting',people:[]},yaw=-0.72,pitch=0.38,zoom=1,panX=0,panY=0,drag=null;
function resize(){const d=devicePixelRatio||1,w=innerWidth,h=innerHeight;canvas.width=Math.round(w*d);canvas.height=Math.round(h*d);canvas.style.width=w+'px';canvas.style.height=h+'px';ctx.setTransform(d,0,0,d,0,0);draw()}
function project(p){const c=Math.cos(yaw),s=Math.sin(yaw),gx=c*p[0]-s*p[1],gd=s*p[0]+c*p[1],scale=Math.min(innerWidth,innerHeight)/8*zoom;return [innerWidth*.5+panX+gx*scale,innerHeight*.62+panY+(gd*Math.sin(pitch)-p[2]*Math.cos(pitch))*scale]}
function line(a,b,color,width=1){const p=project(a),q=project(b);ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke()}
function label(text,p,color){const q=project(p);ctx.fillStyle=color;ctx.font='12px system-ui';ctx.fillText(text,q[0]+6,q[1]-5)}
function grid(){for(let v=-5;v<=5.001;v+=.5){const major=Math.abs(v-Math.round(v))<1e-6;line([v,-5,0],[v,5,0],major?'#29374a':'#182333',major?1.1:.7);line([-5,v,0],[5,v,0],major?'#29374a':'#182333',major?1.1:.7)}line([0,0,0],[1.2,0,0],'#ff6262',3);line([0,0,0],[0,1.2,0],'#62e883',3);line([0,0,0],[0,0,1.2],'#61b8ff',3);label('+X',[1.2,0,0],'#ff7373');label('+Y',[0,1.2,0],'#65e88a');label('+Z',[0,0,1.2],'#67bfff')}
function draw(){ctx.clearRect(0,0,innerWidth,innerHeight);const g=ctx.createRadialGradient(innerWidth*.5,innerHeight*.55,20,innerWidth*.5,innerHeight*.55,Math.max(innerWidth,innerHeight));g.addColorStop(0,'#101a29');g.addColorStop(1,'#05080d');ctx.fillStyle=g;ctx.fillRect(0,0,innerWidth,innerHeight);grid();
for(let pi=0;pi<(frame.people||[]).length;pi++){const person=frame.people[pi],color=palette[pi%palette.length];for(const [a,b] of bones){if(person.joints[a]&&person.joints[b])line(person.joints[a],person.joints[b],color,4)}for(let ji=0;ji<person.joints.length;ji++){const p=person.joints[ji];if(!p)continue;const inferred=person.inferred&&person.inferred[ji],q=project(p);ctx.beginPath();ctx.arc(q[0],q[1],inferred?4:4.5,0,Math.PI*2);ctx.fillStyle=inferred?'#ff9f5a':'#ffe86b';ctx.fill();ctx.strokeStyle=inferred?'#ffd0a8':'#101722';ctx.lineWidth=inferred?1.8:1.2;ctx.stroke()}const root=(person.joints[11]&&person.joints[12])?person.joints[11].map((v,i)=>(v+person.joints[12][i])*.5):person.joints.find(Boolean);if(root)label('ID '+person.id,root,color)} }
function reset(){yaw=-.72;pitch=.38;zoom=1;panX=panY=0;draw()}
canvas.addEventListener('contextmenu',e=>e.preventDefault());canvas.addEventListener('mousedown',e=>{drag={x:e.clientX,y:e.clientY,button:e.button};canvas.classList.add('drag')});addEventListener('mouseup',()=>{drag=null;canvas.classList.remove('drag')});addEventListener('mousemove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;if(drag.button===2){panX+=dx;panY+=dy}else{yaw+=dx*.008;pitch=Math.max(.08,Math.min(1.2,pitch+dy*.006))}draw()});canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.25,Math.min(5,zoom*Math.exp(-e.deltaY*.001)));draw()},{passive:false});addEventListener('keydown',e=>{if(e.key==='r'||e.key==='R')reset()});addEventListener('resize',resize);
async function update(){try{const r=await fetch('/api/frame?'+Date.now(),{cache:'no-store'});frame=await r.json();const stale=frame.stale?' · 数据已停止刷新':'';statusEl.className='status '+(frame.stale?'bad':'good');statusEl.textContent=frame.state==='ready'?`Seq ${frame.sequence} · 人体 ${frame.people.length} · 身体关节 ${frame.valid_joints}（推断 ${frame.inferred_joints||0}） · 延迟 ${frame.age_ms.toFixed(0)} ms${stale}`:'等待 Nano 骨架数据…';draw()}catch(e){statusEl.className='status bad';statusEl.textContent='可视化服务读取失败';}setTimeout(update,100)}
resize();update();
</script></body></html>"""


class SnapshotReader:
    """Read atomically replaced NPZ snapshots and expose compact JSON."""

    def __init__(self, runtime_dir: str | Path, stale_after_s: float = 1.0) -> None:
        self.runtime_dir = Path(runtime_dir).expanduser().resolve()
        self.frame_path = self.runtime_dir / "latest_human_observation.npz"
        self.status_path = self.runtime_dir / "status.json"
        self.stale_after_ns = max(0, int(float(stale_after_s) * 1e9))
        self._lock = threading.Lock()
        self._mtime_ns: int | None = None
        self._frame: dict[str, Any] | None = None

    def _read_status(self) -> dict[str, Any]:
        try:
            value = json.loads(self.status_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (FileNotFoundError, OSError, ValueError):
            return {}

    def _reload_if_needed(self) -> None:
        try:
            mtime_ns = self.frame_path.stat().st_mtime_ns
        except FileNotFoundError:
            self._mtime_ns = None
            self._frame = None
            return
        if self._mtime_ns == mtime_ns:
            return
        try:
            with np.load(self.frame_path, allow_pickle=False) as data:
                skeleton = np.asarray(data["skeleton"], dtype=np.float32)
                human_mask = np.asarray(data["human_mask"], dtype=np.bool_)
                joint_mask = np.asarray(data["joint_mask"], dtype=np.bool_)
                joint_inferred_mask = (
                    np.asarray(data["joint_inferred_mask"], dtype=np.bool_)
                    if "joint_inferred_mask" in data.files
                    else np.zeros_like(joint_mask)
                )
                human_ids = np.asarray(data["human_ids"], dtype=np.int64)
                sequence = int(data["sequence"])
                timestamp_ns = int(data["timestamp_ns"])
        except (OSError, ValueError, EOFError, KeyError):
            return
        if skeleton.ndim != 3 or skeleton.shape[1:] != (17, 7):
            return
        if (human_mask.shape != skeleton.shape[:1]
                or joint_mask.shape != skeleton.shape[:2]
                or joint_inferred_mask.shape != skeleton.shape[:2]):
            return
        people = []
        valid_total = 0
        inferred_total = 0
        for slot in np.flatnonzero(human_mask):
            joints: list[list[float] | None] = []
            inferred: list[bool] = []
            for joint_index in range(17):
                valid = bool(joint_mask[slot, joint_index])
                is_inferred = bool(joint_inferred_mask[slot, joint_index]) and valid
                xyz = skeleton[slot, joint_index, :3]
                if valid and np.isfinite(xyz).all():
                    joints.append([round(float(v), 5) for v in xyz])
                    valid_total += 1
                    inferred_total += int(is_inferred)
                else:
                    joints.append(None)
                inferred.append(is_inferred)
            people.append({
                "id": int(human_ids[slot]), "slot": int(slot),
                "joints": joints, "inferred": inferred,
            })
        self._mtime_ns = mtime_ns
        self._frame = {
            "state": "ready",
            "sequence": sequence,
            "timestamp_ns": timestamp_ns,
            "people": people,
            "valid_joints": valid_total,
            "inferred_joints": inferred_total,
            "pose_source": "causal_kinematic_refinement",
            "motion_prior": "upright_walking",
        }

    def payload(self) -> dict[str, Any]:
        with self._lock:
            self._reload_if_needed()
            if self._frame is None:
                return {
                    "schema": "omninxt.skeleton_viewer.v1",
                    "state": "waiting",
                    "frame_id": "base_link",
                    "axes": {"x": "front", "y": "left", "z": "up"},
                    "people": [], "valid_joints": 0, "stale": True,
                    "age_ms": 0.0,
                }
            result = dict(self._frame)
            status = self._read_status()
            receive_ns = status.get("last_receive_wall_ns")
            age_ns = max(0, time.time_ns() - int(receive_ns)) if receive_ns else 0
            result.update({
                "schema": "omninxt.skeleton_viewer.v1",
                "frame_id": "base_link",
                "coordinate_convention": "+X forward, +Y left, +Z up",
                "units": "metre",
                "age_ms": age_ns / 1e6,
                "stale": not receive_ns or age_ns > self.stale_after_ns,
            })
            return result


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], reader: SnapshotReader) -> None:
        self.reader = reader
        super().__init__(address, ViewerHandler)


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "OmniNxtSkeletonViewer/1"

    def _write(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._write(HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/frame":
            payload = self.server.reader.payload()  # type: ignore[attr-defined]
            self._write(json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                        "application/json; charset=utf-8")
        elif path == "/healthz":
            self._write(b'{"ok":true}', "application/json")
        else:
            self._write(b"not found\n", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)

    def log_message(self, _format: str, *args: object) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Display received COCO-17 skeletons")
    parser.add_argument("--host", default=os.getenv("SKELETON_VIEWER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("SKELETON_VIEWER_PORT", "8767")))
    parser.add_argument("--runtime-dir", default=os.getenv("SKELETON_RUNTIME_DIR", str(DEFAULT_RUNTIME_DIR)))
    parser.add_argument("--stale-after", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reader = SnapshotReader(args.runtime_dir, stale_after_s=args.stale_after)
    server = ViewerServer((args.host, args.port), reader)

    def stop(_signum: int, _frame: object) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    host, port = server.server_address
    print(f"[SKELETON_VIEW] http://{host}:{port} | XY ground, +Z up", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        print("[SKELETON_VIEW] stopped", flush=True)


if __name__ == "__main__":
    main()

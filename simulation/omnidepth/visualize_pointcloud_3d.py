#!/usr/bin/env python3
"""Create a self-contained interactive HTML and a static PNG from an ASCII PCD."""

import argparse
import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_pcd(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line.strip().lower() == "data ascii":
                break
        else:
            raise ValueError("Only ASCII PCD files are supported")
    points = np.loadtxt(path, skiprows=line_no + 1, usecols=(0, 1, 2), dtype=np.float32)
    points = np.atleast_2d(points)
    return points[np.isfinite(points).all(axis=1)]


def sample(points: np.ndarray, maximum: int) -> np.ndarray:
    if len(points) <= maximum:
        return points
    indices = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
    return points[indices]


def save_png(points: np.ndarray, path: Path) -> None:
    pts = sample(points, 50000)
    lo, hi = np.percentile(pts, [1, 99], axis=0)
    keep = np.all((pts >= lo) & (pts <= hi), axis=1)
    pts = pts[keep]
    fig = plt.figure(figsize=(12, 9), dpi=160, facecolor="#080d18")
    ax = fig.add_subplot(111, projection="3d", facecolor="#080d18")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2], cmap="turbo", s=0.35, alpha=0.8)
    ax.scatter([0], [0], [0], c="white", s=28, marker="^")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(f"Omni-Depth 3D point cloud — {len(points):,} points", color="white")
    ax.tick_params(colors="#b8c5d8")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color("#b8c5d8")
        axis.pane.set_facecolor((0.04, 0.07, 0.12, 1.0))
        axis.pane.set_edgecolor((0.2, 0.3, 0.4, 0.5))
    span = np.max(hi - lo); mid = (hi + lo) / 2
    ax.set_xlim(mid[0]-span/2, mid[0]+span/2); ax.set_ylim(mid[1]-span/2, mid[1]+span/2)
    ax.set_zlim(mid[2]-span/2, mid[2]+span/2)
    ax.view_init(elev=28, azim=-55)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def save_html(points: np.ndarray, path: Path, maximum: int) -> None:
    pts = sample(points, maximum)
    center = np.median(pts, axis=0)
    pts = pts - center
    scale = float(np.percentile(np.linalg.norm(pts, axis=1), 98)) or 1.0
    packed = base64.b64encode(pts.astype("<f4").tobytes()).decode("ascii")
    template = r'''<!doctype html><html><head><meta charset="utf-8"><title>Omni-Depth 3D</title>
<style>html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#070b13;color:#dce8ff;font:14px sans-serif}canvas{width:100%;height:100%;display:block}.hud{position:fixed;left:18px;top:16px;padding:12px 15px;background:#0b1324cc;border:1px solid #263b60;border-radius:8px;line-height:1.55}.hud b{font-size:16px}.hint{color:#91a7c8}</style></head>
<body><canvas id="c"></canvas><div class="hud"><b>Omni-Depth 3D point cloud</b><br>POINT_COUNT points<br><span class="hint">左键旋转 · 滚轮缩放 · 右键平移 · R 重置</span></div>
<script>
const raw=atob('POINT_DATA'), buf=new ArrayBuffer(raw.length), u8=new Uint8Array(buf); for(let i=0;i<raw.length;i++)u8[i]=raw.charCodeAt(i); const xyz=new Float32Array(buf);
const c=document.getElementById('c'),gl=c.getContext('webgl',{antialias:true}); if(!gl)alert('浏览器不支持 WebGL');
const vs=`attribute vec3 p;uniform mat4 m;varying float h;void main(){vec4 q=m*vec4(p,1.);gl_Position=q;gl_PointSize=max(1.2,3.0/q.w);h=clamp(p.z/SCALE*.5+.5,0.,1.);}`;
const fs=`precision mediump float;varying float h;vec3 turbo(float x){return clamp(vec3(1.5-abs(4.*x-3.),1.5-abs(4.*x-2.),1.5-abs(4.*x-1.)),0.,1.);}void main(){gl_FragColor=vec4(turbo(h),.9);}`;
function sh(t,s){let x=gl.createShader(t);gl.shaderSource(x,s);gl.compileShader(x);return x}let pr=gl.createProgram();gl.attachShader(pr,sh(gl.VERTEX_SHADER,vs));gl.attachShader(pr,sh(gl.FRAGMENT_SHADER,fs));gl.linkProgram(pr);gl.useProgram(pr);
let b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,xyz,gl.STATIC_DRAW);let p=gl.getAttribLocation(pr,'p');gl.enableVertexAttribArray(p);gl.vertexAttribPointer(p,3,gl.FLOAT,false,0,0);let ml=gl.getUniformLocation(pr,'m');
function mul(a,b){let o=new Float32Array(16);for(let r=0;r<4;r++)for(let q=0;q<4;q++)for(let k=0;k<4;k++)o[q*4+r]+=a[k*4+r]*b[q*4+k];return o}function ident(){return new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1])}function rotX(a){let m=ident(),s=Math.sin(a),q=Math.cos(a);m[5]=q;m[6]=s;m[9]=-s;m[10]=q;return m}function rotZ(a){let m=ident(),s=Math.sin(a),q=Math.cos(a);m[0]=q;m[1]=s;m[4]=-s;m[5]=q;return m}
let yaw=-.7,pitch=.7,zoom=2.7,pan=[0,0],drag=false,last=[0,0],button=0;function reset(){yaw=-.7;pitch=.7;zoom=2.7;pan=[0,0]}c.oncontextmenu=e=>e.preventDefault();c.onmousedown=e=>{drag=true;last=[e.clientX,e.clientY];button=e.button};onmouseup=()=>drag=false;onmousemove=e=>{if(!drag)return;let dx=e.clientX-last[0],dy=e.clientY-last[1];last=[e.clientX,e.clientY];if(button===2){pan[0]+=dx*.002;pan[1]-=dy*.002}else{yaw+=dx*.006;pitch=Math.max(-1.5,Math.min(1.5,pitch+dy*.006))}};c.onwheel=e=>{e.preventDefault();zoom*=Math.exp(e.deltaY*.001)};onkeydown=e=>{if(e.key.toLowerCase()==='r')reset()};
function draw(){let d=devicePixelRatio||1,w=c.clientWidth*d,h=c.clientHeight*d;if(c.width!==w||c.height!==h){c.width=w;c.height=h}gl.viewport(0,0,w,h);gl.clearColor(.027,.043,.075,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);let asp=w/h,f=1.8,P=new Float32Array([f/asp,0,0,0,0,f,0,0,0,0,-1.002,-1,0,0,-.02,0]),M=mul(rotX(pitch),rotZ(yaw));M[12]=pan[0];M[13]=pan[1];M[14]=-zoom;for(let i=0;i<16;i++)if(i<12)M[i]/=SCALE;gl.uniformMatrix4fv(ml,false,mul(P,M));gl.drawArrays(gl.POINTS,0,xyz.length/3);requestAnimationFrame(draw)}draw();
</script></body></html>'''
    html = template.replace("POINT_DATA", packed).replace("POINT_COUNT", f"{len(pts):,}").replace("SCALE", repr(scale))
    path.write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pcd", type=Path)
    ap.add_argument("--max-points", type=int, default=80000)
    args = ap.parse_args()
    points = load_pcd(args.pcd)
    html = args.pcd.with_name(args.pcd.stem + "_3d.html")
    png = args.pcd.with_name(args.pcd.stem + "_3d.png")
    save_html(points, html, args.max_points); save_png(points, png)
    print(f"points={len(points)}\nhtml={html}\npng={png}")


if __name__ == "__main__":
    main()

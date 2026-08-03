#!/usr/bin/env python3
"""Convert an ASCII XYZ/XYZRGB PCD into a standalone interactive HTML."""

import argparse
import base64
import json
import math
import os
import struct


def read_ascii_pcd(path):
    fields = []
    data_start = None
    rows = []
    with open(path, encoding="ascii") as stream:
        lines = stream.readlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.upper().startswith("FIELDS "):
            fields = stripped.split()[1:]
        if stripped.lower() == "data ascii":
            data_start = index + 1
            break
    if data_start is None or not {"x", "y", "z"}.issubset(fields):
        raise ValueError("Expected an ASCII PCD containing x/y/z fields")
    indices = {name: fields.index(name) for name in fields}
    colors = []
    for line in lines[data_start:]:
        values = line.split()
        if len(values) < len(fields):
            continue
        xyz = tuple(float(values[indices[name]]) for name in ("x", "y", "z"))
        if not all(math.isfinite(value) for value in xyz):
            continue
        rows.append(xyz)
        if "rgb" in indices:
            raw = values[indices["rgb"]]
            if any(token in raw.lower() for token in (".", "e")):
                packed = struct.unpack("<I", struct.pack("<f", float(raw)))[0]
            else:
                packed = int(raw)
            colors.extend(((packed >> 16) & 255, (packed >> 8) & 255, packed & 255))
    if not rows:
        raise ValueError("No finite points found")
    points_binary = b"".join(struct.pack("<fff", *point) for point in rows)
    return (
        base64.b64encode(points_binary).decode("ascii"),
        base64.b64encode(bytes(colors)).decode("ascii") if colors else "",
        len(rows),
    )


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OmniDepth 点云查看器</title>
<style>
:root{color-scheme:dark;font-family:Inter,"Noto Sans SC",system-ui,sans-serif}
*{box-sizing:border-box}body{margin:0;overflow:hidden;background:#080b10;color:#e8edf5}
#canvas{position:fixed;inset:0;width:100vw;height:100vh;cursor:grab}
#canvas.drag{cursor:grabbing}.panel{position:fixed;z-index:2;background:rgba(15,20,28,.91);
backdrop-filter:blur(12px);border:1px solid #2c3543;border-radius:12px;box-shadow:0 12px 35px #0008}
#top{left:16px;top:16px;padding:14px 16px;width:min(440px,calc(100vw - 32px))}
h1{font-size:17px;margin:0 0 5px}.sub{font-size:12px;color:#93a2b7;margin-bottom:12px}
.grid{display:grid;grid-template-columns:110px 1fr 52px;gap:8px 10px;align-items:center;font-size:12px}
input[type=range]{width:100%;accent-color:#5ac8fa}select,button,.file{border:1px solid #394657;
background:#18202b;color:#e8edf5;border-radius:7px;padding:6px 9px;font-size:12px}
.actions{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}button,.file{cursor:pointer}
.sector-filter{display:flex;align-items:center;gap:8px 13px;flex-wrap:wrap;margin-top:12px;
padding-top:10px;border-top:1px solid #2c3543;font-size:12px}
.sector-filter strong{color:#aab6c7}.sector-filter label{display:flex;align-items:center;gap:5px;cursor:pointer}
.sector-filter input{margin:0}.sector-filter button{padding:4px 7px}
.file input{display:none}#legend{right:16px;bottom:16px;padding:10px 12px;font-size:12px;line-height:1.55}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
#status{position:fixed;left:16px;bottom:16px;color:#93a2b7;font-size:12px;
background:#0d121acc;border:1px solid #273140;border-radius:8px;padding:7px 10px}
#reference{right:16px;top:16px;width:min(62vw,1050px);padding:9px;display:none}
#reference.open{display:block}#reference img{display:block;width:100%;height:auto;border-radius:7px}
#reference .caption{font-size:11px;color:#aab6c7;padding:7px 3px 1px}
@media(max-width:700px){#legend{display:none}#top{padding:11px}.grid{grid-template-columns:95px 1fr 44px}}
</style>
</head>
<body>
<canvas id="canvas"></canvas>
<section id="top" class="panel">
  <h1>OmniDepth 交互式点云</h1>
  <div class="sub" id="source"></div>
  <div class="grid">
    <label for="size">点大小</label><input id="size" type="range" min=".5" max="5" step=".25" value="1.5"><output id="sizeOut">1.5</output>
    <label for="range">最大距离</label><input id="range" type="range" min=".5" max="10" step=".1" value="6"><output id="rangeOut">6.0m</output>
    <label for="mode">着色</label>
    <select id="mode"><option value="pcd">相机扇区</option><option value="range">距离</option><option value="height">高度</option></select><span></span>
  </div>
  <div class="actions">
    <button id="reset">重置视角 (R)</button>
    <button id="toggleAxes">隐藏坐标轴/网格</button>
    <button id="shot">保存截图</button>
    <button id="toggleImage">显示/隐藏同帧图像</button>
    <label class="file">打开其他ASCII PCD<input id="file" type="file" accept=".pcd"></label>
  </div>
  <div class="sector-filter">
    <strong>显示扇区</strong>
    <label style="color:#ff6868"><input type="checkbox" data-sector="ab" checked>A–B 右侧</label>
    <label style="color:#5ee183"><input type="checkbox" data-sector="bc" checked>B–C 后侧</label>
    <label style="color:#68a0ff"><input type="checkbox" data-sector="cd" checked>C–D 左侧</label>
    <label style="color:#ffe76a"><input type="checkbox" data-sector="da" checked>D–A 前侧</label>
    <button id="allSectors" type="button">全选</button>
    <button id="noSectors" type="button">清空</button>
    <span id="visibleCount"></span>
  </div>
</section>
<aside id="reference" class="panel">
  <img id="referenceImage" src="__IMAGE__" alt="同帧四相机拼接图">
  <div class="caption">与该点云严格匹配的 CAM_A｜CAM_B｜CAM_C｜CAM_D 图像</div>
</aside>
<section id="legend" class="panel">
  <div><span class="dot" style="background:#ff4b4b"></span>A–B 右侧</div>
  <div><span class="dot" style="background:#45db70"></span>B–C 后侧</div>
  <div><span class="dot" style="background:#4b85ff"></span>C–D 左侧</div>
  <div><span class="dot" style="background:#ffe04b"></span>D–A 前侧</div>
</section>
<div id="status">左键旋转 · 滚轮缩放 · 右键/Shift拖动平移</div>
<script>
const embedded={points:"__POINTS__",colors:"__COLORS__",count:__COUNT__,name:__NAME__};
const canvas=document.querySelector("#canvas"),ctx=canvas.getContext("2d",{alpha:false});
let points,colors,count=0,rotX=-.45,rotY=-.7,zoom=1,panX=0,panY=0,drag=false,lastX=0,lastY=0,panMode=false,showAxes=true;
let center=[0,0,0],baseScale=1,dirty=true;
const sectorEnabled={ab:true,bc:true,cd:true,da:true};
const ui={size:document.querySelector("#size"),range:document.querySelector("#range"),
 mode:document.querySelector("#mode"),sizeOut:document.querySelector("#sizeOut"),
 rangeOut:document.querySelector("#rangeOut"),status:document.querySelector("#status"),
 visibleCount:document.querySelector("#visibleCount")};
function decodeFloat32(s){const b=atob(s),a=new Uint8Array(b.length);for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);return new Float32Array(a.buffer)}
function decodeBytes(s){const b=atob(s),a=new Uint8Array(b.length);for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);return a}
function install(p,c,name){points=p;colors=c;count=p.length/3;document.querySelector("#source").textContent=`${name} · ${count.toLocaleString()} 点 · 单文件离线查看`;
 const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
 for(let i=0;i<p.length;i+=3)for(let k=0;k<3;k++){lo[k]=Math.min(lo[k],p[i+k]);hi[k]=Math.max(hi[k],p[i+k])}
 center=lo.map((v,k)=>(v+hi[k])/2);baseScale=1/Math.max(...hi.map((v,k)=>v-lo[k]),.001);reset();dirty=true}
function reset(){rotX=-.45;rotY=-.7;zoom=1;panX=0;panY=0;dirty=true}
function turbo(t){t=Math.max(0,Math.min(1,t));const r=Math.round(255*Math.max(0,Math.min(1,1.5-Math.abs(4*t-3))));
 const g=Math.round(255*Math.max(0,Math.min(1,1.5-Math.abs(4*t-2))));
 const b=Math.round(255*Math.max(0,Math.min(1,1.5-Math.abs(4*t-1))));return [r,g,b]}
function resize(){const d=Math.min(devicePixelRatio,2);canvas.width=innerWidth*d;canvas.height=innerHeight*d;dirty=true}
function sectorOf(j){if(!colors||colors.length<j*3+3)return null;const r=colors[j*3],g=colors[j*3+1],b=colors[j*3+2];
 if(r>200&&g<80&&b<80)return"ab";if(r<80&&g>200&&b<80)return"bc";
 if(r<80&&g<80&&b>200)return"cd";if(r>200&&g>200&&b<80)return"da";return null}
function render(){if(!dirty){requestAnimationFrame(render);return}dirty=false;const w=canvas.width,h=canvas.height;
 ctx.fillStyle="#080b10";ctx.fillRect(0,0,w,h);const sx=Math.sin(rotX),cx=Math.cos(rotX),sy=Math.sin(rotY),cy=Math.cos(rotY);
 const scale=Math.min(w,h)*baseScale*.72*zoom,maxR=+ui.range.value,size=+ui.size.value*devicePixelRatio;
 function project(wx,wy,wz){const x=wx-center[0],y=wy-center[1],z=wz-center[2],x1=cy*x+sy*z,z1=-sy*x+cy*z;
  const y1=cx*y-sx*z1,z2=sx*y+cx*z1,perspective=1/(1+Math.max(-.7,z2*baseScale*.35));
  return [w/2+panX*devicePixelRatio+x1*scale*perspective,h/2+panY*devicePixelRatio-y1*scale*perspective,z2,perspective]}
 if(showAxes){
  ctx.lineWidth=Math.max(1,devicePixelRatio*.7);ctx.strokeStyle="rgba(130,145,165,.18)";
  const gridExtent=Math.max(1,Math.ceil(maxR)),gridStep=gridExtent>6?1:.5;
  for(let g=-gridExtent;g<=gridExtent+.001;g+=gridStep){let a=project(g,-gridExtent,0),b=project(g,gridExtent,0);
   ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();a=project(-gridExtent,g,0);b=project(gridExtent,g,0);
   ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke()}
 }
 const projected=[];
 for(let i=0,j=0;i<points.length;i+=3,j++){let x=points[i]-center[0],y=points[i+1]-center[1],z=points[i+2]-center[2];
  const sector=sectorOf(j);if(sector&&!sectorEnabled[sector])continue;
  const range=Math.hypot(points[i],points[i+1],points[i+2]);if(range>maxR)continue;
  const x1=cy*x+sy*z,z1=-sy*x+cy*z,y1=cx*y-sx*z1,z2=sx*y+cx*z1;
  const perspective=1/(1+Math.max(-.7,z2*baseScale*.35));const px=w/2+panX*devicePixelRatio+x1*scale*perspective;
  const py=h/2+panY*devicePixelRatio-y1*scale*perspective;let c;
  if(ui.mode.value==="pcd"&&colors&&colors.length){c=[colors[j*3],colors[j*3+1],colors[j*3+2]]}
  else if(ui.mode.value==="height")c=turbo((points[i+2]+2)/4);else c=turbo(range/maxR);
  projected.push([z2,px,py,size*Math.max(.55,Math.min(1.8,perspective)),c])}
 ui.visibleCount.textContent=`当前 ${projected.length.toLocaleString()} 点`;
 projected.sort((a,b)=>a[0]-b[0]);
 for(const p of projected){ctx.fillStyle=`rgb(${p[4][0]},${p[4][1]},${p[4][2]})`;ctx.globalAlpha=.35+.55*Math.min(1.2,p[3]/size);
  ctx.fillRect(p[1],p[2],p[3],p[3])}ctx.globalAlpha=1;
 if(showAxes){
  const origin=project(0,0,0),axisLength=Math.min(2,Math.max(.5,maxR/4));
  function axis(end,color,label){const p=project(...end);ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=3*devicePixelRatio;
   ctx.beginPath();ctx.moveTo(origin[0],origin[1]);ctx.lineTo(p[0],p[1]);ctx.stroke();
   const angle=Math.atan2(p[1]-origin[1],p[0]-origin[0]),arrow=9*devicePixelRatio;
   ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(p[0]-arrow*Math.cos(angle-.45),p[1]-arrow*Math.sin(angle-.45));
   ctx.lineTo(p[0]-arrow*Math.cos(angle+.45),p[1]-arrow*Math.sin(angle+.45));ctx.closePath();ctx.fill();
   ctx.font=`bold ${13*devicePixelRatio}px system-ui`;ctx.fillText(label,p[0]+6*devicePixelRatio,p[1]-6*devicePixelRatio)}
  axis([axisLength,0,0],"#ff4b4b",`+X ${axisLength.toFixed(1)}m`);
  axis([0,axisLength,0],"#45db70",`+Y ${axisLength.toFixed(1)}m`);
  axis([0,0,axisLength],"#4b85ff",`+Z ${axisLength.toFixed(1)}m`);
  ctx.fillStyle="#fff";ctx.beginPath();ctx.arc(origin[0],origin[1],4*devicePixelRatio,0,Math.PI*2);ctx.fill();
  ctx.font=`bold ${12*devicePixelRatio}px system-ui`;ctx.fillText("O / IMU",origin[0]+7*devicePixelRatio,origin[1]+15*devicePixelRatio);
  const triadOrigin=[w-76*devicePixelRatio,h-155*devicePixelRatio],triadLength=45*devicePixelRatio;
  function triad(v,color,label){const x=cy*v[0]+sy*v[2],z=-sy*v[0]+cy*v[2],y=cx*v[1]-sx*z;
   const ex=triadOrigin[0]+x*triadLength,ey=triadOrigin[1]-y*triadLength;ctx.strokeStyle=color;ctx.fillStyle=color;
   ctx.lineWidth=3*devicePixelRatio;ctx.beginPath();ctx.moveTo(...triadOrigin);ctx.lineTo(ex,ey);ctx.stroke();
   ctx.font=`bold ${12*devicePixelRatio}px system-ui`;ctx.fillText(label,ex+4*devicePixelRatio,ey-4*devicePixelRatio)}
  triad([1,0,0],"#ff4b4b","X");triad([0,1,0],"#45db70","Y");triad([0,0,1],"#4b85ff","Z");
  ctx.fillStyle="#d9e2ef";ctx.font=`${10*devicePixelRatio}px system-ui`;ctx.fillText("IMU方向",triadOrigin[0]-24*devicePixelRatio,triadOrigin[1]+22*devicePixelRatio)
 }
 requestAnimationFrame(render)}
canvas.onpointerdown=e=>{drag=true;lastX=e.clientX;lastY=e.clientY;panMode=e.button===2||e.shiftKey;canvas.classList.add("drag");canvas.setPointerCapture(e.pointerId)};
canvas.onpointermove=e=>{if(!drag)return;const dx=e.clientX-lastX,dy=e.clientY-lastY;lastX=e.clientX;lastY=e.clientY;
 if(panMode){panX+=dx;panY+=dy}else{rotY+=dx*.007;rotX+=dy*.007}dirty=true};
canvas.onpointerup=()=>{drag=false;canvas.classList.remove("drag")};canvas.oncontextmenu=e=>e.preventDefault();
canvas.onwheel=e=>{e.preventDefault();zoom*=Math.exp(-e.deltaY*.001);zoom=Math.max(.15,Math.min(12,zoom));dirty=true};
for(const el of [ui.size,ui.range,ui.mode])el.oninput=()=>{ui.sizeOut.value=(+ui.size.value).toFixed(1);
 ui.rangeOut.value=(+ui.range.value).toFixed(1)+"m";dirty=true};
document.querySelector("#reset").onclick=reset;addEventListener("keydown",e=>{if(e.key.toLowerCase()==="r")reset()});
document.querySelector("#toggleAxes").onclick=e=>{showAxes=!showAxes;e.target.textContent=showAxes?"隐藏坐标轴/网格":"显示坐标轴/网格";dirty=true};
const sectorChecks=[...document.querySelectorAll("input[data-sector]")];
for(const checkbox of sectorChecks)checkbox.onchange=()=>{sectorEnabled[checkbox.dataset.sector]=checkbox.checked;dirty=true};
function setAllSectors(enabled){for(const checkbox of sectorChecks){checkbox.checked=enabled;sectorEnabled[checkbox.dataset.sector]=enabled}dirty=true}
document.querySelector("#allSectors").onclick=()=>setAllSectors(true);
document.querySelector("#noSectors").onclick=()=>setAllSectors(false);
document.querySelector("#shot").onclick=()=>{const a=document.createElement("a");a.download="omnidepth_pointcloud.png";a.href=canvas.toDataURL("image/png");a.click()};
const reference=document.querySelector("#reference"),referenceImage=document.querySelector("#referenceImage");
if(!referenceImage.getAttribute("src"))document.querySelector("#toggleImage").style.display="none";
document.querySelector("#toggleImage").onclick=()=>reference.classList.toggle("open");
document.querySelector("#file").onchange=async e=>{const file=e.target.files[0];if(!file)return;try{const text=await file.text(),lines=text.split(/\r?\n/);
 let fields=[],start=-1;for(let i=0;i<lines.length;i++){if(lines[i].startsWith("FIELDS "))fields=lines[i].trim().split(/\s+/).slice(1);
 if(lines[i].trim().toLowerCase()==="data ascii"){start=i+1;break}}if(start<0)throw Error("仅支持DATA ascii");
 const xi=fields.indexOf("x"),yi=fields.indexOf("y"),zi=fields.indexOf("z"),ri=fields.indexOf("rgb"),p=[],c=[];
 for(let i=start;i<lines.length;i++){const v=lines[i].trim().split(/\s+/);if(v.length<fields.length)continue;const xyz=[+v[xi],+v[yi],+v[zi]];
 if(!xyz.every(Number.isFinite))continue;p.push(...xyz);if(ri>=0){const n=Number(v[ri])>>>0;c.push((n>>>16)&255,(n>>>8)&255,n&255)}}
 install(new Float32Array(p),c.length?new Uint8Array(c):null,file.name)}catch(err){alert("读取失败："+err.message)}};
addEventListener("resize",resize);resize();install(decodeFloat32(embedded.points),embedded.colors?decodeBytes(embedded.colors):null,embedded.name);render();
</script>
</body></html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pcd")
    parser.add_argument("html")
    parser.add_argument("--image", help="Reference image to embed in the HTML")
    args = parser.parse_args()
    source = os.path.realpath(args.pcd)
    output = os.path.realpath(args.html)
    points, colors, count = read_ascii_pcd(source)
    image_uri = ""
    if args.image:
        image_path = os.path.realpath(args.image)
        extension = os.path.splitext(image_path)[1].lower()
        mime = "image/jpeg" if extension in (".jpg", ".jpeg") else "image/png"
        with open(image_path, "rb") as stream:
            image_uri = (
                f"data:{mime};base64,"
                + base64.b64encode(stream.read()).decode("ascii")
            )
    document = (
        HTML.replace("__POINTS__", points)
        .replace("__COLORS__", colors)
        .replace("__COUNT__", str(count))
        .replace("__NAME__", json.dumps(os.path.basename(source), ensure_ascii=False))
        .replace("__IMAGE__", image_uri)
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        stream.write(document)
    print(output)


if __name__ == "__main__":
    main()

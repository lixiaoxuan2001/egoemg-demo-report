#!/usr/bin/env python3
"""Render the full predicted left-hand sequence and angle plots (no smoothing)."""
from __future__ import annotations
import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG = "#07111f"
PANEL = "#101f31"
LINE = "#263b51"
WHITE = "#f0f6fc"
MUTED = "#9fb2c8"
AMBER = "#f8c16b"
COLORS = ["#f2b66d", "#7ddcc3", "#67b9ef", "#a6a0f5", "#ed93b5"]
CHAINS = [[5,6,7,0], [5,8,9,10,1], [5,11,12,13,2], [5,14,15,16,3], [5,17,18,19,4]]


def font_path(bold=False):
    candidates = (["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if bold else
                  ["/System/Library/Fonts/Supplemental/Arial.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"])
    return next((p for p in candidates if Path(p).is_file()), None)


def plot_angles(folder):
    a = np.load(folder/"predictions.npz", allow_pickle=False)
    t, y = a["time_s"], a["angles_deg"]
    fig, axes = plt.subplots(3,2,figsize=(16,10),layout="constrained")
    names = ["Thumb", "Index", "Middle", "Ring", "Little", "Wrist"]
    for g,ax in enumerate(axes.flat):
        channels = range(g*4,g*4+4) if g<5 else [20,21]
        for j in channels:
            ax.plot(t,y[:,j],lw=1,label=str(a["joint_names"][j]))
        ax.axvspan(0,.255,color="#7e8b99",alpha=.18,label="No prediction" if g==5 else None)
        for lo,hi in [(0,.5),(9.584,10.584),(19.672,20.172)]:
            ax.axvspan(lo,hi,color="#eea93c",alpha=.10)
        ax.axvline(10.084,color="#a16820",lw=1,ls="--")
        transitions = sorted(set((a["window_start_sample"][1:]+510).tolist()) |
                             set((a["window_start_sample"][:-1]+8000).tolist()))
        for s in np.asarray(transitions)/2000:
            ax.axvline(s,color="#93a2b3",lw=.6,alpha=.4)
        ax.set(title=names[g],xlabel="Recording time (s)",ylabel="Predicted angle (degrees)",xlim=(0,20.172))
        ax.legend(fontsize=7,loc="upper right"); ax.grid(alpha=.16)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Custom left-hand EMG / EgoEMG EMGFormer-S\nExploratory predictions; gray lines = window transitions, dashed amber = source join",fontsize=16)
    fig.savefig(folder/"joint_angles.png",dpi=150); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir",type=Path,required=True)
    p.add_argument("--ffmpeg",default="ffmpeg")
    args=p.parse_args(); folder=args.input_dir
    a=np.load(folder/"pose_video_data.npz",allow_pickle=False)
    manifest=json.loads((folder/"manifest.json").read_text())
    transitions=[v["time_s"] for v in manifest["window_transitions"]]
    poses=a["landmarks_mm"]; valid=a["valid"]; times=a["time_s"]
    # UmeTrack fingers extend along +x; +z is the thumb side of this LEFT
    # template. Both camera maps are rotations/projections, never reflections.
    cameras=[np.array([[0,.5,.8660254],[-1,0,0]]),np.array([[0,0,1],[-1,0,0]])]
    boxes=[(60,230,1135,875),(1165,230,1860,650)]
    transforms=[]
    for camera,box in zip(cameras,boxes):
        projected=poses[valid]@camera.T
        lo=np.min(projected,axis=(0,1)); hi=np.max(projected,axis=(0,1))
        center=(hi+lo)/2
        width,height=box[2]-box[0]-130,box[3]-box[1]-125
        scale=min(width/max(hi[0]-lo[0],1),height/max(hi[1]-lo[1],1))*.93
        screen_center=np.array([(box[0]+box[2])/2,(box[1]+box[3])/2+25])
        transforms.append((camera,center,scale,screen_center))
    fonts={size:ImageFont.truetype(font_path(size>=38),size) if font_path(size>=38) else ImageFont.load_default() for size in [18,22,24,28,32,38,48,64]}
    background=Image.new("RGB",(1920,1080),BG); d=ImageDraw.Draw(background)
    d.text((60,38),"EGOEMG  /  CUSTOM RECORDING",fill=COLORS[1],font=fonts[22])
    d.text((60,82),"Left-hand pose prediction",fill=WHITE,font=fonts[48])
    d.text((60,151),"EMGFormer-S   |   8 channels   |   2,000 Hz   |   20.172 s recording",fill=MUTED,font=fonts[24])
    for box in boxes: d.rounded_rectangle(box,radius=24,fill=PANEL,outline=LINE,width=2)
    d.text((90,255),"OBLIQUE VIEW",fill=MUTED,font=fonts[22])
    d.text((1195,255),"PALM-PLANE VIEW",fill=MUTED,font=fonts[22])
    d.text((90,834),"Canonical left hand / fixed camera and scale",fill=MUTED,font=fonts[18])
    d.rounded_rectangle((1165,675,1860,875),radius=24,fill=PANEL,outline=LINE,width=2)
    for j,name in enumerate(["Thumb","Index","Middle","Ring","Little"]):
        x=1195+(j%3)*210; y=701+(j//3)*42
        d.ellipse((x,y+5,x+12,y+17),fill=COLORS[j]);d.text((x+24,y),name,fill=WHITE,font=fonts[22])
    d.text((1195,800),"Wrist angles",fill=MUTED,font=fonts[22])
    d.text((60,1024),"Exploratory output: units and channel mapping unverified; no ground-truth pose or accuracy measurement.",fill=MUTED,font=fonts[22])
    d.text((60,989),"Gray: unavailable / window markers    Amber: input caution / source join    Window changes may cause pose jumps.",fill=MUTED,font=fonts[18])
    cmd=[args.ffmpeg,"-hide_banner","-loglevel","warning","-y","-f","rawvideo","-pix_fmt","rgb24","-s","1920x1080","-r","30","-i","-","-an","-c:v","libx264","-preset","medium","-crf","19","-pix_fmt","yuv420p","-movflags","+faststart",str(folder/"left_hand_pose.mp4")]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    qa_indices=sorted(set([0,8,120,294,303,310,449,562,598,605]))
    (folder/"qa_frames").mkdir(exist_ok=True)
    try:
        for i,t in enumerate(times):
            im=background.copy(); d=ImageDraw.Draw(im)
            d.text((1520,65),f"{t:05.2f} s",fill=WHITE,font=fonts[64])
            d.text((1520,142),"1x playback  /  LEFT",fill=MUTED,font=fonts[22])
            if valid[i]:
                for camera,center,scale,screen_center in transforms:
                    pts=(poses[i]@camera.T-center)*scale+screen_center
                    coords=[tuple(v) for v in pts]
                    d.polygon([coords[k] for k in [5,8,11,14,17]],fill="#1d354b",outline="#53718b",width=3)
                    for chain,color in zip(CHAINS,COLORS):
                        d.line([coords[k] for k in chain],fill=color,width=11,joint="curve")
                        for k in chain[1:]:
                            x,y=coords[k];r=8
                            d.ellipse((x-r,y-r,x+r,y+r),fill=color,outline=WHITE,width=2)
                    x,y=coords[5];d.ellipse((x-10,y-10,x+10,y+10),fill=WHITE)
                wrist=a["angles_deg"][i,20:]
                d.text((1370,800),f"FE {wrist[0]:+.1f} deg   DEV {wrist[1]:+.1f} deg",fill=WHITE,font=fonts[22])
                status="INPUT BOUNDARY CAUTION" if a["boundary_caution"][i] else "PREDICTED POSE"
                color=AMBER if a["boundary_caution"][i] else COLORS[1]
                if any(abs(t-s)<=.10 for s in transitions):
                    status="WINDOW TRANSITION / POSE MAY JUMP"; color=AMBER
            else:
                d.text((260,480),"No prediction yet",fill=MUTED,font=fonts[48])
                d.text((260,550),"Initial 0.255 s is model context",fill=MUTED,font=fonts[28])
                status="NO PREDICTION / INITIAL CONTEXT"; color=MUTED
            d.text((60,192),status,fill=color,font=fonts[22])
            x0,x1,y=60,1860,931
            d.rounded_rectangle((x0,y,x1,y+15),radius=7,fill=LINE)
            for lo,hi in [(0,.5),(9.584,10.584),(19.672,20.172)]:
                d.rectangle((x0+(x1-x0)*lo/20.172,y,x0+(x1-x0)*hi/20.172,y+15),fill="#775a35")
            d.rectangle((x0,y,x0+(x1-x0)*.255/20.172,y+15),fill="#536276")
            join=x0+(x1-x0)*10.084/20.172
            d.line((join,y-10,join,y+25),fill=AMBER,width=2)
            for boundary in transitions:
                bx=x0+(x1-x0)*boundary/20.172
                d.line((bx,y-5,bx,y+20),fill=MUTED,width=1)
            x=x0+(x1-x0)*t/20.172
            d.ellipse((x-10,y-3,x+10,y+18),fill=WHITE)
            d.text((60,955),"0 s",fill=MUTED,font=fonts[18]);d.text((1770,955),"20.172 s",fill=MUTED,font=fonts[18])
            if i in qa_indices: im.save(folder/"qa_frames"/f"frame_{i:04}.png")
            if i==120: im.save(folder/"video_poster.jpg",quality=92)
            proc.stdin.write(im.tobytes())
            if i%150==0: print(f"Rendered {i}/{len(times)} frames",flush=True)
    finally:
        proc.stdin.close()
    if proc.wait()!=0: raise RuntimeError("ffmpeg failed")
    thumbs=[]
    for i in qa_indices:
        thumb=Image.open(folder/"qa_frames"/f"frame_{i:04}.png").resize((640,360))
        thumbs.append(thumb)
    sheet=Image.new("RGB",(1280,360*math.ceil(len(thumbs)/2)),BG)
    for j,thumb in enumerate(thumbs): sheet.paste(thumb,((j%2)*640,(j//2)*360))
    sheet.save(folder/"video_contact_sheet.jpg",quality=90)
    plot_angles(folder)
    web=[]
    for i,t in enumerate(times):
        web.append({"t":round(float(t),6),"valid":bool(valid[i]),"caution":bool(a["boundary_caution"][i]),"angles_deg":np.round(a["angles_deg"][i],4).tolist() if valid[i] else None})
    (folder/"viewer_data.json").write_text(json.dumps({"joint_names":manifest["joint_names"],"duration_s":20.172,"window_transitions_s":transitions,"frames":web},separators=(",",":"),allow_nan=False))
    print("Video and angle plots complete",flush=True)


if __name__=="__main__": main()

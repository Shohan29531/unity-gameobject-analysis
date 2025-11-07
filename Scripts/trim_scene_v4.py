#!/usr/bin/env python3
"""
trim_scene_v5.py — Generalized 3D path summarizer (scene-neutral, mid-detail text).

Inputs:
  --static  path to worldGenerated_static.csv
  --frames  path to worldGenerated_frames.csv

Behavior:
  - Uses all frames (no temporal subsampling).
  - Computes one compact, data-derived natural language summary per object.
  - No flight metaphors, no frame-by-frame narration.
  - Drops MovementSummary entirely.

Outputs:
  trajectories.csv with:
      ObjectName
      ObjectType (from static file, if available)
      StartFrame
      EndFrame
      StartPosition
      EndPosition
      AvgSpeed
      VisibilityRatio
      SizeChange
      RelativeSize
      PathSummary   ← main column used for LLM input
"""

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- helpers ----------

NUM_RE = re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')


def parse_vec3(s):
    """Parse (x,y,z) style strings into floats."""
    if pd.isna(s):
        return (np.nan, np.nan, np.nan)
    nums = NUM_RE.findall(str(s))
    if len(nums) >= 3:
        return (float(nums[0]), float(nums[1]), float(nums[2]))
    return (np.nan, np.nan, np.nan)


def vec_len(dx, dy, dz):
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def classify_size_change(volumes):
    vols = np.array([v for v in volumes if v > 0 and np.isfinite(v)])
    if vols.size < 4:
        return "unknown"
    k = max(1, vols.size // 4)
    start_med = float(np.median(vols[:k]))
    end_med = float(np.median(vols[-k:]))
    if start_med <= 0 or end_med <= 0:
        return "unknown"
    ratio = end_med / start_med
    if ratio > 1.3:
        return "grow"
    if ratio < 0.77:
        return "shrink"
    return "stable"


def classify_relative_size(object_med_vol, global_median):
    if (
        not np.isfinite(object_med_vol)
        or object_med_vol <= 0
        or not np.isfinite(global_median)
        or global_median <= 0
    ):
        return "unknown"
    r = object_med_vol / global_median
    if r < 0.3:
        return "tiny"
    if r < 0.7:
        return "small"
    if r < 1.5:
        return "medium"
    if r < 3.0:
        return "large"
    return "huge"


# ---------- mid-detail motion summarizer ----------

def speed_word(v):
    if v < 1e-4:
        return "barely moves"
    if v < 0.02:
        return "moves very slowly"
    if v < 0.08:
        return "moves slowly"
    if v < 0.2:
        return "moves at a moderate pace"
    return "moves quickly"


def summarize_motion(frames, px, py, pz):
    """Generate mid-detail, scene-neutral summary for an object's motion."""
    n = len(frames)
    if n < 2:
        return "The object shows no significant motion."

    dx = np.diff(px)
    dy = np.diff(py)
    dz = np.diff(pz)
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    total_path = np.nansum(dist)
    mean_speed = np.nanmean(dist)
    vert_range = np.nanmax(py) - np.nanmin(py)
    horiz_disp = math.sqrt((px[-1] - px[0])**2 + (pz[-1] - pz[0])**2)

    # Motion classification
    motion_type = "stationary"
    if total_path > 0.3:
        if vert_range > horiz_disp * 1.2:
            motion_type = "primarily vertical"
        elif horiz_disp > vert_range * 1.2:
            motion_type = "primarily horizontal"
        else:
            motion_type = "multi-directional"

    # Direction cues
    vert_dir = "upward" if py[-1] > py[0] + 0.2 else "downward" if py[-1] < py[0] - 0.2 else "level"
    horiz_dir = []
    if pz[-1] < pz[0] - 0.2:
        horiz_dir.append("forward")
    if pz[-1] > pz[0] + 0.2:
        horiz_dir.append("backward")
    if px[-1] > px[0] + 0.2:
        horiz_dir.append("to the right")
    if px[-1] < px[0] - 0.2:
        horiz_dir.append("to the left")
    horiz_phrase = " and ".join(horiz_dir) if horiz_dir else "without strong horizontal bias"

    # Qualitative speed
    speed_desc = speed_word(mean_speed)

    parts = [
        f"The object {speed_desc} along a {motion_type} path.",
        f"It travels roughly {total_path:.2f} world units, moving {vert_dir} and {horiz_phrase}.",
    ]
    if vert_range > 0.5:
        parts.append(f"It spans about {vert_range:.2f} units vertically.")
    if horiz_disp > 0.5:
        parts.append(f"Its horizontal displacement from start to end is about {horiz_disp:.2f} units.")

    return " ".join(parts)


# ---------- main ----------

def parse_args():
    p = argparse.ArgumentParser(description="Generate trajectories.csv with mid-detail, scene-neutral motion summaries.")
    p.add_argument("--static", required=True)
    p.add_argument("--frames", required=True)
    p.add_argument("--outdir", default=None)
    p.add_argument("--movement-threshold", type=float, default=0.2)
    p.add_argument("--vis-threshold", type=float, default=0.05)
    p.add_argument("--top-k", type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()

    frames_df = pd.read_csv(args.frames)
    static_df = pd.read_csv(args.static)

    required_cols = ["Frame", "ObjectName", "Position", "RendererSize", "ColliderSize"]
    missing = [c for c in required_cols if c not in frames_df.columns]
    if missing:
        raise ValueError(f"Frames file missing required columns: {missing}")

    # Merge static info
    static_cols = [c for c in ["ObjectName", "Type", "Tag", "Level", "Active", "InView",
                               "HasRenderer", "HasCollider"] if c in static_df.columns]
    if "ObjectName" not in static_cols:
        static_cols = ["ObjectName"]
    static_slim = static_df[static_cols].drop_duplicates(subset=["ObjectName"], keep="first")
    frames_df = frames_df.merge(static_slim, on="ObjectName", how="left", suffixes=("", "_static"))

    # Parse positions and sizes
    pos = np.vstack([parse_vec3(v) for v in frames_df["Position"]])
    frames_df["pos_x"], frames_df["pos_y"], frames_df["pos_z"] = pos[:, 0], pos[:, 1], pos[:, 2]

    rend = np.vstack([parse_vec3(v) for v in frames_df.get("RendererSize", pd.Series([None]*len(frames_df)))])
    coll = np.vstack([parse_vec3(v) for v in frames_df.get("ColliderSize", pd.Series([None]*len(frames_df)))])
    rend_vol = np.abs(rend[:, 0]*rend[:, 1]*rend[:, 2])
    coll_vol = np.abs(coll[:, 0]*coll[:, 1]*coll[:, 2])
    frames_df["Volume"] = np.where(np.isfinite(rend_vol) & (rend_vol > 0), rend_vol,
                                   np.where(np.isfinite(coll_vol) & (coll_vol > 0), coll_vol, np.nan))

    frames_df = frames_df.sort_values(["ObjectName", "Frame"])
    obj_med_vol = frames_df.groupby("ObjectName")["Volume"].median()
    global_med_vol = float(obj_med_vol[obj_med_vol > 0].median()) if (obj_med_vol > 0).any() else np.nan

    records = []
    for obj, g in frames_df.groupby("ObjectName"):
        g = g.sort_values("Frame")
        frames = g["Frame"].to_numpy()
        if len(frames) == 0:
            continue

        px, py, pz = g["pos_x"].to_numpy(), g["pos_y"].to_numpy(), g["pos_z"].to_numpy()
        vol_series = g["Volume"].to_numpy()
        valid_pos = np.isfinite(px) & np.isfinite(py) & np.isfinite(pz)

        start_frame, end_frame = int(frames[0]), int(frames[-1])
        vis_ratio = float(np.isfinite(vol_series).sum() / max(1, len(g)))
        size_change = classify_size_change(vol_series)
        med_vol = float(obj_med_vol.get(obj, np.nan))
        rel_size = classify_relative_size(med_vol, global_med_vol)
        obj_type = g["Type"].iloc[0] if "Type" in g.columns else ""

        idx_valid = np.where(valid_pos)[0]
        if idx_valid.size < 2:
            continue

        first_idx, last_idx = idx_valid[0], idx_valid[-1]
        start_pos = (px[first_idx], py[first_idx], pz[first_idx])
        end_pos = (px[last_idx], py[last_idx], pz[last_idx])

        speeds, total_path = [], 0.0
        for i in range(1, len(frames)):
            if not (valid_pos[i] and valid_pos[i - 1]):
                continue
            dt = frames[i] - frames[i - 1]
            if dt <= 0:
                continue
            dx, dy, dz = px[i] - px[i - 1], py[i] - py[i - 1], pz[i] - pz[i - 1]
            dist = vec_len(dx, dy, dz)
            total_path += dist
            speeds.append(dist / dt)
        avg_speed = float(np.mean(speeds)) if speeds else 0.0

        # Build summary
        path_summary = summarize_motion(frames, px, py, pz)

        # Importance for top-k selection
        size_weight = {"tiny":0.5, "small":0.8, "medium":1.0, "large":1.4, "huge":1.8}.get(rel_size, 0.7)
        importance = (total_path + 0.05 * vec_len(end_pos[0]-start_pos[0],
                                                 end_pos[1]-start_pos[1],
                                                 end_pos[2]-start_pos[2])) * (0.5 + vis_ratio) * size_weight

        records.append({
            "ObjectName": obj,
            "ObjectType (from static file, if available)": obj_type,
            "StartFrame": start_frame,
            "EndFrame": end_frame,
            "StartPosition": f"[{start_pos[0]},{start_pos[1]},{start_pos[2]}]",
            "EndPosition": f"[{end_pos[0]},{end_pos[1]},{end_pos[2]}]",
            "AvgSpeed": avg_speed,
            "VisibilityRatio": vis_ratio,
            "SizeChange": size_change,
            "RelativeSize": rel_size,
            "PathSummary": path_summary,
            "_ImportanceScore": importance,
            "_TotalPath": total_path,
        })

    if not records:
        print("No trajectories produced.")
        return

    df = pd.DataFrame(records)

    # Filtering
    move_mask = df["_TotalPath"] >= args.movement_threshold
    vis_mask = df["VisibilityRatio"] >= args.vis_threshold
    size_mask = df["RelativeSize"].isin(["large", "huge"])
    df = df[move_mask | vis_mask | size_mask]
    if df.empty:
        print("No trajectories left after filtering.")
        return

    df = df.sort_values("_ImportanceScore", ascending=False)
    if args.top_k and args.top_k > 0 and len(df) > args.top_k:
        df = df.head(args.top_k)
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")

    outdir = Path(args.outdir) if args.outdir else (Path(args.frames).parent / "trajectories_v5")
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "trajectories.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Wrote {out_path} with {len(df)} summarized trajectories.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
trim_scene_v6.py — Hierarchical motion summarizer for Unity scenes.

Goal:
  Produce a CSV (trajectories.csv) where each object's motion is described
  as a compact, data-driven PathSummary suitable for feeding to an LLM.

Summary format:
  Overall: [global summary].
  Phases: [phase 1]; [phase 2]; [phase 3].

Behavior:
  - Scene-neutral (no "takeoff"/"landing"/"flying" language)
  - Identifies 2–4 major motion phases using changes in altitude,
    horizontal displacement, and speed.
  - Keeps language concise but descriptive (≈2–4 sentences per object)
  - Drops MovementSummary entirely.

Outputs (trajectories.csv):
  ObjectName, ObjectType, Start/End frames & positions,
  AvgSpeed, VisibilityRatio, SizeChange, RelativeSize, PathSummary
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


# ---------- summarization ----------

def speed_word(v):
    if v < 0.02:
        return "very slowly"
    if v < 0.08:
        return "slowly"
    if v < 0.2:
        return "at a moderate speed"
    return "quickly"


def summarize_hierarchical(frames, px, py, pz):
    """Generate a hierarchical PathSummary: global + phase sketch."""
    n = len(frames)
    if n < 2:
        return "Overall: no significant motion detected."

    dx = np.diff(px)
    dy = np.diff(py)
    dz = np.diff(pz)
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    total_path = np.nansum(dist)
    avg_speed = np.nanmean(dist)
    vert_range = np.nanmax(py) - np.nanmin(py)
    horiz_disp = math.sqrt((px[-1]-px[0])**2 + (pz[-1]-pz[0])**2)

    # ----- GLOBAL OVERVIEW -----
    if total_path < 0.2:
        overall = "Overall: remains mostly stationary with minimal displacement."
    else:
        distance_label = (
            "short" if total_path < 1.0 else
            "moderate" if total_path < 5.0 else "long"
        )
        vert_label = (
            "flat" if vert_range < 0.5 else
            "moderate" if vert_range < 2.0 else "large"
        )
        motion_type = (
            "primarily vertical" if vert_range > horiz_disp * 1.3 else
            "primarily horizontal" if horiz_disp > vert_range * 1.3 else
            "multi-directional"
        )
        overall = (
            f"Overall: moves {speed_word(avg_speed)} along a {distance_label}, "
            f"{motion_type} path with {vert_label} vertical variation."
        )

    # ----- PHASE EXTRACTION -----
    # Simple heuristic segmentation: detect altitude trend changes
    # Simple heuristic segmentation: detect altitude trend changes
    dy_smooth = pd.Series(dy).rolling(5, min_periods=1, center=True).mean()
    dy_vals = dy_smooth.to_numpy()
    sign_changes = np.where(np.sign(dy_vals[:-1]) != np.sign(dy_vals[1:]))[0]

    key_idxs = sorted(set([0, len(py)//3, 2*len(py)//3, len(py)-1] + list(sign_changes)))
    key_idxs = [i for i in key_idxs if 0 <= i < len(py)]
    if len(key_idxs) > 6:
        key_idxs = key_idxs[:6]

    phases = []
    for i in range(1, len(key_idxs)):
        s, e = key_idxs[i-1], key_idxs[i]
        dy_seg = py[e] - py[s]
        horiz = math.sqrt((px[e]-px[s])**2 + (pz[e]-pz[s])**2)
        seg_len = math.sqrt(dy_seg**2 + horiz**2)
        if seg_len < 0.05:
            phases.append("remains nearly still")
            continue
        vert_phrase = (
            "ascends" if dy_seg > 0.3 else
            "descends" if dy_seg < -0.3 else
            "maintains level altitude"
        )
        horiz_phrase = (
            "moves across the scene" if horiz > 0.3 else
            "shifts slightly"
        )
        if vert_phrase == "maintains level altitude":
            phases.append(horiz_phrase)
        else:
            phases.append(f"{vert_phrase} and {horiz_phrase}")

    # Compress duplicates
    final_phrases = []
    for p in phases:
        if not final_phrases or final_phrases[-1] != p:
            final_phrases.append(p)

    # Limit to 3–4 phrases
    final_phrases = final_phrases[:4]
    phase_text = "; ".join(final_phrases)
    if not phase_text:
        phase_text = "shows no distinct motion phases."

    return f"{overall} Phases: {phase_text}."


# ---------- main ----------

def parse_args():
    p = argparse.ArgumentParser(description="Generate trajectories.csv with hierarchical motion summaries.")
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
    static_cols = [c for c in ["ObjectName", "Type", "Tag", "Level", "Active",
                               "InView", "HasRenderer", "HasCollider"]
                   if c in static_df.columns]
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
        idx_valid = np.where(valid_pos)[0]
        if idx_valid.size < 2:
            continue

        start_frame, end_frame = int(frames[0]), int(frames[-1])
        vis_ratio = float(np.isfinite(vol_series).sum() / max(1, len(g)))
        size_change = classify_size_change(vol_series)
        med_vol = float(obj_med_vol.get(obj, np.nan))
        rel_size = classify_relative_size(med_vol, global_med_vol)
        obj_type = g["Type"].iloc[0] if "Type" in g.columns else ""

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

        path_summary = summarize_hierarchical(frames, px, py, pz)
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

    outdir = Path(args.outdir) if args.outdir else (Path(args.frames).parent / "trajectories_v6")
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "trajectories.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Wrote {out_path} with {len(df)} hierarchical motion summaries.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
trim_scene_v3.py — Trajectory-level scene compressor for LLM narration.

Inputs:
  --static  path to worldGenerated_static.csv
  --frames  path to worldGenerated_frames.csv

This script:
  - Reads the ORIGINAL frames file (must include):
        Frame, ObjectName, Position, RotationEuler, RotationQuat,
        RendererCenter, RendererSize, ColliderCenter, ColliderSize
  - Optionally uses static file metadata (Type, Level, Active, InView, etc.)
  - Computes, for each important object:
        * Where it starts / moves / ends
        * Straight vs curved vs jittery path
        * Speed, acceleration, and their dominant directions
        * Rotation behavior
        * Visibility and size evolution
        * Relative size vs the rest of the scene
        * A short natural language MovementSummary

Output:
  trajectories.csv — ONE ROW PER OBJECT with LLM-friendly columns.

Example:
  python trim_scene_v3.py \
    --static "../Data/Slime Rancher/worldGenerated_static.csv" \
    --frames "../Data/Slime Rancher/worldGenerated_frames.csv" \
    --movement-threshold 0.2 \
    --vis-threshold 0.05 \
    --top-k 300
"""

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

# --------- Regex & parsing helpers ---------

NUM_RE = re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')


def parse_vec3(s):
    """Parse first 3 numeric values from a string like '(x,y,z)' or '[x y z]'.
    Returns (x,y,z) as floats or (nan,nan,nan) if invalid."""
    if pd.isna(s):
        return (np.nan, np.nan, np.nan)
    text = str(s)
    nums = NUM_RE.findall(text)
    if len(nums) >= 3:
        return (float(nums[0]), float(nums[1]), float(nums[2]))
    return (np.nan, np.nan, np.nan)


def vec_len(dx, dy, dz):
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def coarse_direction(vec, eps=1e-6):
    """Map a 3D vector to a coarse textual direction for LLMs."""
    x, y, z = vec
    mag = vec_len(x, y, z)
    if not np.isfinite(mag) or mag < eps:
        return "none"

    # Normalize
    x /= mag
    y /= mag
    z /= mag

    # Basic axis contributions
    parts = []

    # Forward/back (use -z as forward if using Unity-style)
    if z < -0.25:
        parts.append("forward")
    elif z > 0.25:
        parts.append("backward")

    # Left/right (x)
    if x > 0.25:
        parts.append("right")
    elif x < -0.25:
        parts.append("left")

    # Up/down (y)
    if y > 0.25:
        parts.append("upward")
    elif y < -0.25:
        parts.append("downward")

    if not parts:
        return "slight"

    return "-".join(parts)


def classify_path_type(straightness, net_disp, total_path):
    """Heuristic classification of motion pattern."""
    if total_path < 1e-4 and net_disp < 1e-4:
        return "static"
    if straightness >= 0.9:
        return "straight"
    if straightness >= 0.6:
        return "curved"
    # Low straightness: distinguish oscillate vs jitter
    if net_disp < 0.25 * total_path:
        return "oscillate"
    return "jitter"


def classify_rotation_pattern(yaw_diffs):
    """Very rough rotation behavior from yaw deltas (in degrees)."""
    if len(yaw_diffs) == 0:
        return "none"

    total_abs = float(np.nansum(np.abs(yaw_diffs)))
    net = float(np.nansum(yaw_diffs))

    if total_abs < 5.0:
        return "none"
    if total_abs > 360.0:
        return "spin"
    # Mostly one-direction
    if abs(net) > 0.7 * total_abs:
        return "slow turn right" if net < 0 else "slow turn left"
    # Otherwise wiggly
    return "oscillate"


def classify_size_change(volumes):
    """Classify grow/shrink/stable from per-frame volumes."""
    vols = np.array([v for v in volumes if v > 0 and np.isfinite(v)])
    if vols.size < 4:
        return "unknown"
    # Use early vs late medians
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
    if not np.isfinite(object_med_vol) or object_med_vol <= 0 or not np.isfinite(global_median) or global_median <= 0:
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


def movement_summary_row(row):
    """Build a short natural language summary from row fields."""
    size = row.get("RelativeSize (approx. size compared to average scene object)", "unknown")
    path_type = row.get("PathType (motion classification)", "static")
    dom_dir = row.get("DominantDirection (overall movement direction)", "none")
    rot = row.get("RotationPattern (qualitative rotation)", "none")
    avg_speed = row.get("AvgSpeed (avg world units per frame)", 0.0)
    avg_accel = row.get("AvgAcceleration (avg change in speed per frame)", 0.0)

    # Size phrase
    if size in ("large", "huge"):
        prefix = "Large object"
    elif size in ("tiny", "small"):
        prefix = "Small object"
    else:
        prefix = "Object"

    # Motion phrase
    if path_type == "static" or (avg_speed < 1e-4):
        motion = "remains mostly still"
    else:
        speed_word = "slowly"
        if avg_speed > 0.2:
            speed_word = "quickly"
        elif avg_speed > 0.05:
            speed_word = "steadily"

        dir_word = "" if dom_dir in ("none", "slight") else f" {dom_dir}"
        motion = f"moves {speed_word}{dir_word}".strip()

        if avg_accel > 0.05:
            motion += " with noticeable acceleration"
        elif avg_accel > 0.01:
            motion += " with slight acceleration"

    # Rotation phrase
    rot_phrase = ""
    if rot == "none":
        rot_phrase = ""
    elif rot.startswith("slow turn"):
        rot_phrase = f" while {rot}"
    elif rot in ("spin", "oscillate"):
        rot_phrase = f" and {rot}s"
    else:
        rot_phrase = f" with {rot}"

    sentence = f"{prefix} {motion}{rot_phrase}."
    # Cleanup double spaces
    sentence = " ".join(sentence.split())
    return sentence


# --------- Arg parsing ---------

def parse_args():
    p = argparse.ArgumentParser(description="Generate compact trajectories.csv for LLM narration.")
    p.add_argument("--static", required=True, help="Path to worldGenerated_static.csv")
    p.add_argument("--frames", required=True, help="Path to worldGenerated_frames.csv (original, full columns)")
    p.add_argument("--outdir", default=None, help="Output dir (default: frames dir /trajectories_v3)")
    p.add_argument("--movement-threshold", type=float, default=0.2,
                   help="Min total path length to consider object important")
    p.add_argument("--vis-threshold", type=float, default=0.02,
                   help="Min visibility ratio to consider object important")
    p.add_argument("--top-k", type=int, default=300,
                   help="Keep only top-K objects by importance (0 = keep all)")
    return p.parse_args()


# --------- Main ---------

def main():
    args = parse_args()

    frames_df = pd.read_csv(args.frames)
    static_df = pd.read_csv(args.static)

    # Basic sanity
    required_cols = [
        "Frame", "ObjectName", "Position", "RotationEuler",
        "RendererCenter", "RendererSize", "ColliderCenter", "ColliderSize"
    ]
    missing = [c for c in required_cols if c not in frames_df.columns]
    if missing:
        raise ValueError(f"Frames file missing required columns: {missing}")

    # Merge static info (if columns exist)
    static_cols_possible = ["ObjectName", "Type", "Tag", "Level",
                            "Active", "InView", "HasRenderer", "HasCollider"]
    static_cols = [c for c in static_cols_possible if c in static_df.columns]
    if "ObjectName" not in static_cols:
        static_cols = ["ObjectName"]
    static_slim = static_df[static_cols].drop_duplicates(subset=["ObjectName"], keep="first")

    frames_df = frames_df.merge(static_slim, on="ObjectName", how="left", suffixes=("", "_static"))

    # Parse positions and volumes
    pos_xyz = np.vstack([parse_vec3(v) for v in frames_df["Position"]])
    frames_df["pos_x"] = pos_xyz[:, 0]
    frames_df["pos_y"] = pos_xyz[:, 1]
    frames_df["pos_z"] = pos_xyz[:, 2]

    # Renderer volume
    rend_xyz = np.vstack([parse_vec3(v) for v in frames_df["RendererSize"]])
    rend_vol = np.abs(rend_xyz[:, 0] * rend_xyz[:, 1] * rend_xyz[:, 2])

    # Collider volume
    coll_xyz = np.vstack([parse_vec3(v) for v in frames_df["ColliderSize"]])
    coll_vol = np.abs(coll_xyz[:, 0] * coll_xyz[:, 1] * coll_xyz[:, 2])

    # Prefer renderer volume, else collider
    vol = np.where(np.isfinite(rend_vol) & (rend_vol > 0), rend_vol,
                   np.where(np.isfinite(coll_vol) & (coll_vol > 0), coll_vol, np.nan))
    frames_df["Volume"] = vol

    # Sort by object & frame
    frames_df = frames_df.sort_values(["ObjectName", "Frame"])

    # Compute median volume per object for RelativeSize
    obj_med_vol = frames_df.groupby("ObjectName")["Volume"].median()
    global_med_vol = float(obj_med_vol[obj_med_vol > 0].median()) if (obj_med_vol > 0).any() else np.nan

    # Prepare per-object trajectories
    records = []
    for obj, g in frames_df.groupby("ObjectName"):
        g = g.sort_values("Frame")
        frames = g["Frame"].to_numpy()

        # Skip objects with no frames
        if len(frames) == 0:
            continue

        # Positions
        px = g["pos_x"].to_numpy()
        py = g["pos_y"].to_numpy()
        pz = g["pos_z"].to_numpy()

        # Valid position mask
        valid_pos = np.isfinite(px) & np.isfinite(py) & np.isfinite(pz)
        if not valid_pos.any():
            # No usable movement, but might still be visible/large/static
            # We'll handle via visibility/size.
            start_frame = int(frames[0])
            end_frame = int(frames[-1])
            vis_ratio = float(np.isfinite(g["Volume"]).sum() / max(1, len(g)))
            med_vol = float(obj_med_vol.get(obj, np.nan))
            rel_size = classify_relative_size(med_vol, global_med_vol)
            size_change = classify_size_change(g["Volume"].to_numpy())
            obj_type = g["Type"].iloc[0] if "Type" in g.columns else np.nan

            # Filter static-only low-visibility small junk
            if vis_ratio < args.vis_threshold and rel_size in ("tiny", "unknown"):
                continue

            path_type = "static"
            summary = movement_summary_row({
                "RelativeSize (approx. size compared to average scene object)": rel_size,
                "PathType (motion classification)": path_type,
                "DominantDirection (overall movement direction)": "none",
                "RotationPattern (qualitative rotation)": "none",
                "AvgSpeed (avg world units per frame)": 0.0,
                "AvgAcceleration (avg change in speed per frame)": 0.0,
            })

            records.append({
                "ObjectName": obj,
                "ObjectType (from static file, if available)": obj_type,
                "StartFrame (first frame where object is relevant)": start_frame,
                "EndFrame (last frame where object is relevant)": end_frame,
                "StartPosition (x,y,z at StartFrame)": "unknown",
                "MidPosition (x,y,z at midpoint)": "unknown",
                "EndPosition (x,y,z at EndFrame)": "unknown",
                "PathStraightness (0–1; 1 = straight line)": 0.0,
                "PathType (motion classification)": path_type,
                "AvgSpeed (avg world units per frame)": 0.0,
                "PeakSpeed (maximum world units per frame)": 0.0,
                "AvgAcceleration (avg change in speed per frame)": 0.0,
                "AccelerationDirection (dominant direction of acceleration)": "none",
                "DominantDirection (overall movement direction)": "none",
                "RotationPattern (qualitative rotation)": "none",
                "VisibilityRatio (fraction of frames visible)": vis_ratio,
                "SizeChange (grow/shrink/stable)": size_change,
                "RelativeSize (approx. size compared to average scene object)": rel_size,
                "MovementSummary (natural language summary)": summary,
                "_ImportanceScore": float((vis_ratio + 0.1) * ({"tiny":0.5,"small":0.8,"medium":1,"large":1.3,"huge":1.6}.get(rel_size,0.7)))
            })
            continue

        # Restrict to frames with valid positions for motion metrics
        idx_valid = np.where(valid_pos)[0]
        if idx_valid.size < 2:
            # No real motion, treat like static-visible object
            start_frame = int(frames[0])
            end_frame = int(frames[-1])
            vis_ratio = float(np.isfinite(g["Volume"]).sum() / max(1, len(g)))
            med_vol = float(obj_med_vol.get(obj, np.nan))
            rel_size = classify_relative_size(med_vol, global_med_vol)
            size_change = classify_size_change(g["Volume"].to_numpy())
            obj_type = g["Type"].iloc[0] if "Type" in g.columns else np.nan

            if vis_ratio < args.vis_threshold and rel_size in ("tiny", "unknown"):
                continue

            path_type = "static"
            summary = movement_summary_row({
                "RelativeSize (approx. size compared to average scene object)": rel_size,
                "PathType (motion classification)": path_type,
                "DominantDirection (overall movement direction)": "none",
                "RotationPattern (qualitative rotation)": "none",
                "AvgSpeed (avg world units per frame)": 0.0,
                "AvgAcceleration (avg change in speed per frame)": 0.0,
            })

            records.append({
                "ObjectName": obj,
                "ObjectType (from static file, if available)": obj_type,
                "StartFrame (first frame where object is relevant)": start_frame,
                "EndFrame (last frame where object is relevant)": end_frame,
                "StartPosition (x,y,z at StartFrame)": "unknown",
                "MidPosition (x,y,z at midpoint)": "unknown",
                "EndPosition (x,y,z at EndFrame)": "unknown",
                "PathStraightness (0–1; 1 = straight line)": 0.0,
                "PathType (motion classification)": path_type,
                "AvgSpeed (avg world units per frame)": 0.0,
                "PeakSpeed (maximum world units per frame)": 0.0,
                "AvgAcceleration (avg change in speed per frame)": 0.0,
                "AccelerationDirection (dominant direction of acceleration)": "none",
                "DominantDirection (overall movement direction)": "none",
                "RotationPattern (qualitative rotation)": "none",
                "VisibilityRatio (fraction of frames visible)": vis_ratio,
                "SizeChange (grow/shrink/stable)": size_change,
                "RelativeSize (approx. size compared to average scene object)": rel_size,
                "MovementSummary (natural language summary)": summary,
                "_ImportanceScore": float((vis_ratio + 0.1) * ({"tiny":0.5,"small":0.8,"medium":1,"large":1.3,"huge":1.6}.get(rel_size,0.7)))
            })
            continue

        first_idx = idx_valid[0]
        last_idx = idx_valid[-1]
        start_frame = int(frames[first_idx])
        end_frame = int(frames[last_idx])

        # Key positions
        start_pos = (px[first_idx], py[first_idx], pz[first_idx])
        end_pos = (px[last_idx], py[last_idx], pz[last_idx])
        mid_frame_target = (start_frame + end_frame) / 2.0
        mid_idx = idx_valid[np.argmin(np.abs(frames[idx_valid] - mid_frame_target))]
        mid_pos = (px[mid_idx], py[mid_idx], pz[mid_idx])

        # Speeds & velocities
        speeds = []
        vel_vecs = []
        total_path = 0.0

        last_v = None
        for i in range(first_idx + 1, last_idx + 1):
            if not (valid_pos[i] and valid_pos[i - 1]):
                continue
            dt = frames[i] - frames[i - 1]
            if dt <= 0:
                continue
            dx = px[i] - px[i - 1]
            dy = py[i] - py[i - 1]
            dz = pz[i] - pz[i - 1]
            step_dist = vec_len(dx, dy, dz)
            if step_dist < 0:
                continue
            total_path += step_dist
            speed = step_dist / dt
            vx, vy, vz = dx / dt, dy / dt, dz / dt
            speeds.append(speed)
            vel_vecs.append((vx, vy, vz))
            last_v = (vx, vy, vz)

        net_dx = end_pos[0] - start_pos[0]
        net_dy = end_pos[1] - start_pos[1]
        net_dz = end_pos[2] - start_pos[2]
        net_disp = vec_len(net_dx, net_dy, net_dz)

        if total_path <= 1e-8:
            straightness = 0.0
        else:
            straightness = max(0.0, min(1.0, net_disp / total_path))

        # Acceleration
        acc_mags = []
        acc_vecs = []
        if len(vel_vecs) >= 2:
            # Align with frames (approx: consecutive valid position steps)
            step_indices = [i for i in range(first_idx + 1, last_idx + 1) if valid_pos[i] and valid_pos[i - 1]]
            for j in range(1, len(vel_vecs)):
                vx1, vy1, vz1 = vel_vecs[j - 1]
                vx2, vy2, vz2 = vel_vecs[j]
                f1, f2 = frames[step_indices[j - 1]], frames[step_indices[j]]
                dt = f2 - f1
                if dt <= 0:
                    continue
                ax = (vx2 - vx1) / dt
                ay = (vy2 - vy1) / dt
                az = (vz2 - vz1) / dt
                mag = vec_len(ax, ay, az)
                acc_mags.append(mag)
                acc_vecs.append((ax, ay, az))

        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        peak_speed = float(np.max(speeds)) if speeds else 0.0
        avg_accel = float(np.mean(acc_mags)) if acc_mags else 0.0

        # Dominant directions
        dom_dir = coarse_direction((net_dx, net_dy, net_dz))
        if acc_vecs:
            mean_ax = float(np.mean([v[0] for v in acc_vecs]))
            mean_ay = float(np.mean([v[1] for v in acc_vecs]))
            mean_az = float(np.mean([v[2] for v in acc_vecs]))
            acc_dir = coarse_direction((mean_ax, mean_ay, mean_az))
        else:
            acc_dir = "none"

        # Path type
        path_type = classify_path_type(straightness, net_disp, total_path)

        # Rotation pattern (focus on yaw from RotationEuler if available)
        yaw_diffs = []
        if "RotationEuler" in g.columns:
            rot_xyz = np.vstack([parse_vec3(v) for v in g["RotationEuler"]])
            yaw = rot_xyz[:, 1]  # assume Y is yaw
            yaw_valid = np.isfinite(yaw)
            last_yaw = None
            last_frame_y = None
            for fval, yv in zip(frames, yaw):
                if not np.isfinite(yv):
                    continue
                if last_yaw is not None:
                    dyaw = yv - last_yaw
                    # Normalize to [-180, 180] for wrap-around
                    while dyaw > 180:
                        dyaw -= 360
                    while dyaw < -180:
                        dyaw += 360
                    yaw_diffs.append(dyaw)
                last_yaw = yv
                last_frame_y = fval
        rot_pattern = classify_rotation_pattern(yaw_diffs)

        # Visibility ratio & size metrics
        vol_series = g["Volume"].to_numpy()
        vis_ratio = float(np.isfinite(vol_series).sum() / max(1, len(g)))
        size_change = classify_size_change(vol_series)
        med_vol = float(obj_med_vol.get(obj, np.nan))
        rel_size = classify_relative_size(med_vol, global_med_vol)

        # Object type from static if available
        obj_type = g["Type"].iloc[0] if "Type" in g.columns else np.nan

        # Importance score for filtering
        size_weight = {"tiny": 0.5, "small": 0.8, "medium": 1.0, "large": 1.4, "huge": 1.8}.get(rel_size, 0.7)
        importance = (total_path + 0.05 * net_disp) * (0.5 + vis_ratio) * size_weight




        # Filter out very unimportant objects
        if total_path < args.movement_threshold and vis_ratio < args.vis_threshold and rel_size in ("tiny", "unknown"):
            # NOTE: `args.movement-threshold` is invalid syntax; corrected below in final code.
            pass
        # We'll correct this conditional below.

        # Build summary row proto (fix filter after computing)
        record = {
            "ObjectName": obj,
            "ObjectType (from static file, if available)": obj_type,
            "StartFrame (first frame where object is relevant)": int(start_frame),
            "EndFrame (last frame where object is relevant)": int(end_frame),
            "StartPosition (x,y,z at StartFrame)": f"[{start_pos[0]},{start_pos[1]},{start_pos[2]}]",
            "MidPosition (x,y,z at midpoint)": f"[{mid_pos[0]},{mid_pos[1]},{mid_pos[2]}]",
            "EndPosition (x,y,z at EndFrame)": f"[{end_pos[0]},{end_pos[1]},{end_pos[2]}]",
            "PathStraightness (0–1; 1 = straight line)": float(straightness),
            "PathType (motion classification)": path_type,
            "AvgSpeed (avg world units per frame)": float(avg_speed),
            "PeakSpeed (maximum world units per frame)": float(peak_speed),
            "AvgAcceleration (avg change in speed per frame)": float(avg_accel),
            "AccelerationDirection (dominant direction of acceleration)": acc_dir,
            "DominantDirection (overall movement direction)": dom_dir,
            "RotationPattern (qualitative rotation)": rot_pattern,
            "VisibilityRatio (fraction of frames visible)": float(vis_ratio),
            "SizeChange (grow/shrink/stable)": size_change,
            "RelativeSize (approx. size compared to average scene object)": rel_size,
            "_TotalPath": float(total_path),
            "_NetDisplacement": float(net_disp),
            "_ImportanceScore": float(importance),
        }

        records.append(record)

    # Fix filtering bug from inside loop by applying here
    df = pd.DataFrame(records)
    if df.empty:
        print("No trajectories produced (all objects filtered).")
        return

    # Apply movement & visibility thresholds
    if "_TotalPath" in df.columns and "_NetDisplacement" in df.columns:
        move_mask = (df["_TotalPath"] >= args.movement_threshold) | (df["_NetDisplacement"] >= args.movement_threshold / 2.0)
    else:
        move_mask = True
    vis_mask = df["VisibilityRatio (fraction of frames visible)"] >= args.vis_threshold

    # Keep objects that either move enough OR are visible enough OR are large
    keep_mask = (
        move_mask |
        vis_mask |
        df["RelativeSize (approx. size compared to average scene object)"].isin(["large", "huge"])
    )
    df = df[keep_mask]

    if df.empty:
        print("No trajectories left after applying thresholds.")
        return

    # Recompute movement summaries now that rows are final
    summaries = []
    for _, row in df.iterrows():
        summaries.append(movement_summary_row(row))
    df["MovementSummary (natural language summary)"] = summaries

    # Sort by importance
    if "_ImportanceScore" in df.columns:
        df = df.sort_values("_ImportanceScore", ascending=False)

    # Apply top-k
    if args.top_k and args.top_k > 0 and len(df) > args.top_k:
        df = df.head(args.top_k)

    # Drop internal columns
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")

    # Output
    outdir = Path(args.outdir) if args.outdir else (Path(args.frames).parent / "trajectories_v3")
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "trajectories.csv"
    df.to_csv(out_path, index=False)

    print(f"✅ Wrote {out_path} with {len(df)} object trajectories.")


if __name__ == "__main__":
    main()

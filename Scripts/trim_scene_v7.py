#!/usr/bin/env python3
"""
trim_scene_v7.py — Object-Trajectory Narratives (OTN) extractor.

This script converts Unity engine telemetry (static + frame CSVs)
into a structured, object-centric representation plus a concise text layer.

Design:
  - OTN-Core: identities, hierarchy, lifecycle, motion statistics, phase codes.
  - OTN-Text: PathSummary = deterministic textualization of OTN-Core features.
  - Quantile-based bands (short/medium/long, slow/medium/fast, flat/moderate/large)
    instead of arbitrary numeric thresholds.
  - Simple symbolic motion phases derived from relative motion patterns.
  - Scene-neutral, no application-specific metaphors.

Outputs: trajectories_v7/trajectories.csv with columns:

  # Identity / hierarchy
  ObjectName (Name of the Object)
  ObjectCategory (Potential Object Type)
  # ObjectPath                (commented out for now)
  # ParentName                (commented out for now)
  # Level                     (commented out for now)

  # Lifecycle
  StartFrame (Frame where the object first appears)
  EndFrame (Frame where the object last appears)

  # Core motion stats (OTN-Core)
  StartPosition [x,y,z] at first valid position
  EndPosition [x,y,z] at last valid position
  PathLength (Sum of all displacements, added stepwise with abs value)
  NetDisplacement (Vector value of end position - start position)
  AvgSpeed (Mean step speed over valid steps)
  VisibilityRatio (Percentage of frames where the object is visible)
  SizeChange (Change in perceived size of the object, can be due to camera movement)
  RelativeSize (Compared to other objects in the game)

  # Normalized bands (OTN-Core, discretized)
  DistBand
  SpeedBand
  VertBand

  # Symbolic phases + text (OTN-Text)
  MotionPhases (Symbolic phase sequence, e.g. STILL_LOW;ASCEND_MID)
  PathSummary (Short, deterministic natural-language description)

Filtering:
  - Importance uses PathLength, NetDisplacement, VisibilityRatio, RelativeSize.
  - Apply movement-threshold, vis-threshold, top-k on OTN-Core metrics.

Note:
  Column names in the output CSV are verbose and self-descriptive for LLMs.
  Internally, we use short, stable keys and rename at the end.
"""

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- parsing helpers ----------

NUM_RE = re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')


def parse_vec3(s):
    """Parse first 3 numbers from '(x,y,z)' / '[x y z]' / etc."""
    if pd.isna(s):
        return (np.nan, np.nan, np.nan)
    nums = NUM_RE.findall(str(s))
    if len(nums) >= 3:
        return (float(nums[0]), float(nums[1]), float(nums[2]))
    return (np.nan, np.nan, np.nan)


def vec_len(dx, dy, dz):
    return math.sqrt(dx * dx + dy * dy + dz * dz)


# ---------- size helpers ----------

def classify_size_change(volumes):
    """grow/shrink/stable/unknown based on start vs end median volumes."""
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
    """tiny/small/medium/large/huge/unknown relative to global median volume."""
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


# ---------- quantile-based banding (dataset-relative) ----------

def make_band_fn(values, labels):
    """
    Build a banding function from numeric values using quantiles.

    labels: ordered list, e.g. ["none","short","medium","long"].
    - 'none' is used when value <= 0 or no spread.
    - Remaining labels are split by quantiles (e.g. 3 bands => 2 cutpoints).
    """
    vals = np.array([v for v in values if np.isfinite(v) and v > 0])
    if vals.size < 3 or np.allclose(vals, vals[0]):
        # Degenerate: no meaningful separation; map positives to a middle-ish band.
        def band(v):
            if not np.isfinite(v) or v <= 0:
                return labels[0]
            return labels[-2] if len(labels) > 2 else labels[-1]
        return band

    eff_labels = labels[1:]
    n = len(eff_labels)
    qs = [i / n for i in range(1, n)]
    cuts = np.quantile(vals, qs)

    def band(v):
        if not np.isfinite(v) or v <= 0:
            return labels[0]
        for c, lab in zip(cuts, eff_labels):
            if v <= c:
                return lab
        return eff_labels[-1]

    return band


# ---------- altitude bands (scene-relative) ----------

def make_altitude_band_fn(all_y):
    ys = np.array([y for y in all_y if np.isfinite(y)])
    if ys.size < 3 or np.allclose(ys, ys[0]):
        def f(_y):
            return "MID"
        return f
    q1, q2 = np.quantile(ys, [0.33, 0.66])

    def f(y):
        if not np.isfinite(y):
            return "MID"
        if y <= q1:
            return "LOW"
        if y <= q2:
            return "MID"
        return "HIGH"

    return f


# ---------- motion phase extraction (per object) ----------

def extract_motion_phases(frames, px, py, pz, altitude_band_fn):
    """
    Return a list of symbolic phase codes, e.g.:
      ["STILL_LOW", "ASCEND_LOW", "HORIZ_MID"]

    Principles:
      - STILL: very low speed relative to object's own non-zero speeds.
      - ASCEND/DESCEND: vertical motion dominates and has consistent sign.
      - HORIZ: horizontal motion dominates.
      - MIXED: fallback for non-dominant cases.
      - Each step is tagged with an altitude band (LOW/MID/HIGH) at its midpoint.
      - Consecutive identical codes are merged.
      - Output length is capped to keep things compact.
    """
    n = len(frames)
    if n < 2:
        return []

    dx = np.diff(px)
    dy = np.diff(py)
    dz = np.diff(pz)
    dt = np.diff(frames)

    valid = (
        np.isfinite(dx) & np.isfinite(dy) & np.isfinite(dz) &
        np.isfinite(dt) & (dt > 0)
    )
    if not valid.any():
        return []

    step_speed = np.zeros(n - 1)
    step_alt = np.empty(n - 1)

    for i in range(n - 1):
        if not valid[i]:
            step_speed[i] = 0.0
            step_alt[i] = np.nan
            continue
        d = vec_len(dx[i], dy[i], dz[i])
        step_speed[i] = d / dt[i]
        mid_y = 0.5 * (py[i] + py[i + 1])
        step_alt[i] = mid_y

    non_zero_speeds = step_speed[step_speed > 0]
    if non_zero_speeds.size == 0:
        # Entire trajectory effectively static.
        med_y = float(np.nanmedian(py)) if np.isfinite(np.nanmedian(py)) else np.nan
        alt = altitude_band_fn(med_y)
        return [f"STILL_{alt}"]

    speed_scale = np.median(non_zero_speeds)
    still_thresh = 0.1 * speed_scale  # relative to object motion

    phases = []
    for i in range(n - 1):
        if not valid[i]:
            continue

        s = step_speed[i]
        if not np.isfinite(s):
            continue

        alt_band = altitude_band_fn(step_alt[i])

        horiz = math.sqrt(dx[i] * dx[i] + dz[i] * dz[i])
        v = abs(dy[i])

        if s < still_thresh:
            code = f"STILL_{alt_band}"
        else:
            if v > horiz and dy[i] > 0:
                code = f"ASCEND_{alt_band}"
            elif v > horiz and dy[i] < 0:
                code = f"DESCEND_{alt_band}"
            elif horiz >= v:
                code = f"HORIZ_{alt_band}"
            else:
                code = f"MIXED_{alt_band}"

        if not phases or phases[-1] != code:
            phases.append(code)

    # Cap number of phases
    max_phases = 6
    if len(phases) > max_phases:
        idxs = np.linspace(0, len(phases) - 1, max_phases).astype(int)
        phases = [phases[i] for i in idxs]

    return phases


# ---------- OTN-Text: PathSummary construction ----------

def pathsummary_from_core(stats, phases):
    """
    Build a short, deterministic NL summary from:
      - stats: dict with DistBand, SpeedBand, VertBand
      - phases: list of codes like "HORIZ_LOW", "ASCEND_MID"
    """
    dist_band = stats.get("DistBand", "none")
    speed_band = stats.get("SpeedBand", "none")
    vert_band = stats.get("VertBand", "none")

    speed_word = {
        "none": "stays mostly still",
        "slow": "moves slowly",
        "medium": "moves at a moderate speed",
        "fast": "moves quickly",
    }.get(speed_band, "moves")

    dist_word = {
        "none": "with negligible displacement",
        "short": "over a short distance",
        "medium": "over a medium distance",
        "long": "over a long distance",
    }.get(dist_band, "over some distance")

    vert_word = {
        "none": "",
        "flat": "with almost no vertical change",
        "moderate": "with moderate vertical variation",
        "large": "with large vertical variation",
    }.get(vert_band, "")

    parts = []

    if speed_band == "none" and dist_band == "none":
        parts.append("Overall: stays mostly still.")
    else:
        base = f"Overall: {speed_word} {dist_word}"
        if vert_word:
            base += f" {vert_word}"
        base += "."
        parts.append(base)

    if not phases:
        return " ".join(parts)

    def phase_phrase(code):
        base, _, alt = code.partition("_")
        alt = alt or "MID"

        loc = {
            "LOW": "in a low region",
            "MID": "in a mid-height region",
            "HIGH": "in a high region",
        }.get(alt, "")

        if base == "STILL":
            return "stays mostly still" + (f" {loc}" if loc else "")
        if base == "ASCEND":
            return "moves upward" + (f" {loc}" if loc else "")
        if base == "DESCEND":
            return "moves downward" + (f" {loc}" if loc else "")
        if base == "HORIZ":
            return "moves across the scene" + (f" {loc}" if loc else "")
        if base == "MIXED":
            return "moves with changing direction" + (f" {loc}" if loc else "")
        return None

    phrases = []
    for c in phases:
        p = phase_phrase(c)
        if p and (not phrases or phrases[-1] != p):
            phrases.append(p)

    if not phrases:
        return " ".join(parts)

    max_phrases = 4
    phrases = phrases[:max_phrases]
    phase_text = "; ".join(phrases)
    parts.append(f"Phases: {phase_text}.")

    return " ".join(parts)


# ---------- main ----------

def parse_args():
    p = argparse.ArgumentParser(description="Generate trajectories.csv with OTN-Core + OTN-Text (v7).")
    p.add_argument("--static", required=True)
    p.add_argument("--frames", required=True)
    p.add_argument("--outdir", default=None)
    p.add_argument("--movement-threshold", type=float, default=0.2,
                   help="Min PathLength to consider object for inclusion (world units).")
    p.add_argument("--vis-threshold", type=float, default=0.05,
                   help="Min visibility ratio to consider object.")
    p.add_argument("--top-k", type=int, default=300,
                   help="Keep top-K objects by importance (0 = keep all).")
    return p.parse_args()


def main():
    args = parse_args()

    frames_df = pd.read_csv(args.frames)
    static_df = pd.read_csv(args.static)

    required_cols = ["Frame", "ObjectName", "Position"]
    missing = [c for c in required_cols if c not in frames_df.columns]
    if missing:
        raise ValueError(f"Frames file missing required columns: {missing}")

    # ---- Hierarchy / identity from static ----
    static_cols = static_df.columns

    has_parent = "Parent" in static_cols
    has_level = "Level" in static_cols
    has_type = "Type" in static_cols
    has_layer = "Layer" in static_cols
    has_tag = "Tag" in static_cols

    hier_info = {}
    for _, row in static_df.iterrows():
        name = row.get("ObjectName")
        if pd.isna(name):
            continue
        name = str(name)
        parent = str(row["Parent"]) if has_parent and not pd.isna(row["Parent"]) else ""
        level = int(row["Level"]) if has_level and not pd.isna(row["Level"]) else None
        typ = str(row["Type"]) if has_type and not pd.isna(row["Type"]) else ""
        layer = str(row["Layer"]) if has_layer and not pd.isna(row["Layer"]) else ""
        tag = str(row["Tag"]) if has_tag and not pd.isna(row["Tag"]) else ""

        if "/" in name or not parent:
            path = name
        else:
            path = f"{parent}/{name}"

        cat_candidates = [typ, tag, layer]
        category = next((c for c in cat_candidates if c), "")

        hier_info[name] = {
            "ObjectPath": path,
            "ParentName": parent,
            "Level": level,
            "ObjectCategory": category,
        }

    # ---- Positions & volumes ----
    pos = np.vstack([parse_vec3(v) for v in frames_df["Position"]])
    frames_df["pos_x"], frames_df["pos_y"], frames_df["pos_z"] = pos[:, 0], pos[:, 1], pos[:, 2]

    if "RendererSize" in frames_df.columns:
        rend = np.vstack([parse_vec3(v) for v in frames_df["RendererSize"]])
        rend_vol = np.abs(rend[:, 0] * rend[:, 1] * rend[:, 2])
    else:
        rend_vol = np.full(len(frames_df), np.nan)

    if "ColliderSize" in frames_df.columns:
        coll = np.vstack([parse_vec3(v) for v in frames_df["ColliderSize"]])
        coll_vol = np.abs(coll[:, 0] * coll[:, 1] * coll[:, 2])
    else:
        coll_vol = np.full(len(frames_df), np.nan)

    vol = np.where(np.isfinite(rend_vol) & (rend_vol > 0), rend_vol,
                   np.where(np.isfinite(coll_vol) & (coll_vol > 0), coll_vol, np.nan))
    frames_df["Volume"] = vol

    frames_df = frames_df.sort_values(["ObjectName", "Frame"])

    obj_med_vol = frames_df.groupby("ObjectName")["Volume"].median()
    global_med_vol = float(obj_med_vol[obj_med_vol > 0].median()) if (obj_med_vol > 0).any() else np.nan

    all_y = frames_df["pos_y"].to_numpy()
    altitude_band_fn = make_altitude_band_fn(all_y)

    # ---- First pass: per-object stats ----
    per_obj = []

    for obj, g in frames_df.groupby("ObjectName"):
        g = g.sort_values("Frame")
        frames = g["Frame"].to_numpy()
        px = g["pos_x"].to_numpy()
        py = g["pos_y"].to_numpy()
        pz = g["pos_z"].to_numpy()
        vol_series = g["Volume"].to_numpy()

        valid_pos = np.isfinite(px) & np.isfinite(py) & np.isfinite(pz)
        if valid_pos.sum() < 2:
            continue

        idx_valid = np.where(valid_pos)[0]
        first_idx, last_idx = idx_valid[0], idx_valid[-1]

        # Lifecycle: full span where object appears
        start_frame = int(frames[0])
        end_frame = int(frames[-1])

        start_pos = (px[first_idx], py[first_idx], pz[first_idx])
        end_pos = (px[last_idx], py[last_idx], pz[last_idx])

        vis_ratio = float(np.isfinite(vol_series).sum() / max(1, len(g)))
        size_change = classify_size_change(vol_series)
        med_vol = float(obj_med_vol.get(obj, np.nan))
        rel_size = classify_relative_size(med_vol, global_med_vol)

        dx = np.diff(px)
        dy = np.diff(py)
        dz = np.diff(pz)
        dt = np.diff(frames)

        step_valid = (
            np.isfinite(dx) & np.isfinite(dy) & np.isfinite(dz) &
            np.isfinite(dt) & (dt > 0)
        )
        if not step_valid.any():
            continue

        step_dists = np.zeros(len(dx))
        step_speeds = np.zeros(len(dx))
        for i in range(len(dx)):
            if not step_valid[i]:
                continue
            d = vec_len(dx[i], dy[i], dz[i])
            step_dists[i] = d
            step_speeds[i] = d / dt[i]

        path_length = float(step_dists.sum())
        net_disp = float(vec_len(
            end_pos[0] - start_pos[0],
            end_pos[1] - start_pos[1],
            end_pos[2] - start_pos[2],
        ))
        moving_speeds = step_speeds[step_speeds > 0]
        avg_speed = float(moving_speeds.mean()) if moving_speeds.size > 0 else 0.0

        vert_range = float(np.nanmax(py[idx_valid]) - np.nanmin(py[idx_valid]))
        horiz_extent = float(math.sqrt(
            (np.nanmax(px[idx_valid]) - np.nanmin(px[idx_valid])) ** 2 +
            (np.nanmax(pz[idx_valid]) - np.nanmin(pz[idx_valid])) ** 2
        ))

        phases = extract_motion_phases(frames[idx_valid], px[idx_valid], py[idx_valid], pz[idx_valid],
                                       altitude_band_fn)
        motion_phases_str = ";".join(phases)

        h = hier_info.get(obj, {})
        category = h.get("ObjectCategory", "")

        per_obj.append({
            "ObjectName": obj,
            # "ObjectPath": h.get("ObjectPath", obj),   # kept commented out
            # "ParentName": h.get("ParentName", ""),    # kept commented out
            # "Level": h.get("Level", None),            # kept commented out
            "ObjectCategory": category,
            "StartFrame": start_frame,
            "EndFrame": end_frame,
            "StartPosition": f"[{start_pos[0]},{start_pos[1]},{start_pos[2]}]",
            "EndPosition": f"[{end_pos[0]},{end_pos[1]},{end_pos[2]}]",
            "PathLength": path_length,
            "NetDisplacement": net_disp,
            "AvgSpeed": avg_speed,
            "VisibilityRatio": vis_ratio,
            "SizeChange": size_change,
            "RelativeSize": rel_size,
            "VertRange": vert_range,
            "HorizExtent": horiz_extent,
            "MotionPhases": motion_phases_str,
        })

    if not per_obj:
        print("No trajectories produced.")
        return

    stats_df = pd.DataFrame(per_obj)

    # ---- Band functions ----
    dist_band_fn = make_band_fn(stats_df["PathLength"].tolist(),
                                labels=["none", "short", "medium", "long"])
    speed_band_fn = make_band_fn(stats_df["AvgSpeed"].tolist(),
                                 labels=["none", "slow", "medium", "fast"])
    vert_band_fn = make_band_fn(stats_df["VertRange"].tolist(),
                                labels=["none", "flat", "moderate", "large"])

    # ---- Second pass: PathSummary, importance, filtering ----
    records = []
    for row in per_obj:
        path_length = row["PathLength"]
        avg_speed = row["AvgSpeed"]
        vert_range = row["VertRange"]
        vis_ratio = row["VisibilityRatio"]
        rel_size = row["RelativeSize"]
        net_disp = row["NetDisplacement"]

        dist_band = dist_band_fn(path_length)
        speed_band = speed_band_fn(avg_speed)
        vert_band = vert_band_fn(vert_range)

        phases = row["MotionPhases"].split(";") if row["MotionPhases"] else []
        stats_for_text = {
            "DistBand": dist_band,
            "SpeedBand": speed_band,
            "VertBand": vert_band,
        }
        path_summary = pathsummary_from_core(stats_for_text, phases)

        size_weight = {
            "tiny": 0.5,
            "small": 0.8,
            "medium": 1.0,
            "large": 1.4,
            "huge": 1.8,
        }.get(rel_size, 0.7)

        importance = (
            (path_length + 0.25 * net_disp) *
            (0.5 + vis_ratio) *
            size_weight
        )

        rec = dict(row)
        rec.update({
            "DistBand": dist_band,
            "SpeedBand": speed_band,
            "VertBand": vert_band,
            "PathSummary": path_summary,
            "_ImportanceScore": float(importance),
        })
        records.append(rec)

    df = pd.DataFrame(records)

    # Thresholding
    move_mask = df["PathLength"] >= args.movement_threshold
    vis_mask = df["VisibilityRatio"] >= args.vis_threshold
    size_mask = df["RelativeSize"].isin(["large", "huge"])
    keep_mask = move_mask | vis_mask | size_mask
    df = df[keep_mask]

    if df.empty:
        print("No trajectories left after applying thresholds.")
        return

    # Sort & top-k
    df = df.sort_values("_ImportanceScore", ascending=False)
    if args.top_k and args.top_k > 0 and len(df) > args.top_k:
        df = df.head(args.top_k)

    # Drop internals not intended as final OTN features
    df = df.drop(columns=["VertRange", "HorizExtent", "_ImportanceScore"], errors="ignore")

    # Human-readable column names for final CSV
    rename_map = {
        "ObjectName": "ObjectName (Name of the Object)",
        "ObjectCategory": "ObjectCategory (Potential Object Type)",
        "StartFrame": "StartFrame (Frame where the object first appears)",
        "EndFrame": "EndFrame (Frame where the object last appears)",
        "StartPosition": "StartPosition [x,y,z] at first valid position",
        "EndPosition": "EndPosition [x,y,z] at last valid position",
        "PathLength": "PathLength (Sum of all displacements, added stepwise with abs value)",
        "NetDisplacement": "NetDisplacement (Vector value of end position - start position)",
        "AvgSpeed": "AvgSpeed (Mean step speed over valid steps)",
        "VisibilityRatio": "VisibilityRatio (Percentage of frames where the object is visible)",
        "SizeChange": "SizeChange (Change in perceived size of the object, can be due to camera movement)",
        "RelativeSize": "RelativeSize (Compared to other objects in the game)",
        "MotionPhases": "MotionPhases (Symbolic phase sequence, e.g. STILL_LOW;ASCEND_MID)",
        "PathSummary": "PathSummary (Short, deterministic natural-language description)",
        "DistBand": "DistBand (Categorical range of total distance covered: none/short/medium/long)",
        "SpeedBand": "SpeedBand (Categorical range of average movement speed: none/slow/medium/fast)",
        "VertBand": "VertBand (Categorical range of vertical displacement: none/flat/moderate/large)",
    }
    df = df.rename(columns=rename_map)

    # Output
    outdir = Path(args.outdir) if args.outdir else (Path(args.frames).parent / "trajectories_v7")
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "trajectories.csv"
    df.to_csv(out_path, index=False)

    print(f"✅ Wrote {out_path} with {len(df)} OTN trajectories (v7).")


if __name__ == "__main__":
    main()

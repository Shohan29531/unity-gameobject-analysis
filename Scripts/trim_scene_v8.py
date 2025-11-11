#!/usr/bin/env python3
"""
trim_scene_v8.py — Object-Trajectory Narratives (OTN) extractor, enhanced.

This script converts Unity engine telemetry (static + frame CSVs)
into a structured, object-centric representation plus richer narrative layers.

It extends v7 with:
  - Temporal path traces (downsampled trajectories).
  - Phase timing (relative durations per motion phase).
  - Simple object–object interaction signals.
  - Outlier / notable motion events.
  - Heuristic semantic roles.
  - A higher-level NarrativeSummary while preserving the v7 PathSummary.

Design layers:

  OTN-Core:
    - identities, hierarchy, lifecycle
    - motion statistics & bands (dist/speed/vertical)
    - phase codes

  OTN-Text (per object):
    - PathSummary:
        deterministic NL from core stats + phases (backwards compatible with v7)
    - PathTrace:
        downsampled (frame, position) key points to preserve path shape
    - PhaseTimeline:
        motion phases with approximate time-fractions
    - InteractionSummary:
        close-approach / touch-style hints to other important objects
    - MaxSpeedEvent / LargestVerticalChangeEvent:
        notable local events
    - SemanticRole:
        heuristic high-level role (Player/Enemy/Projectile/etc.)
    - NarrativeSummary:
        compact, chronological, narrative-style description using all above

Outputs:
  trajectories_v8/trajectories.csv

Notes:
  - Column names remain verbose and LLM-friendly.
  - v7 columns are preserved (where applicable) for compatibility.
  - Enhancements are designed to be cheap and scene-agnostic.
"""

import argparse
import math
import re
from collections import defaultdict
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
    - Remaining labels are split by quantiles.
    """
    vals = np.array([v for v in values if np.isfinite(v) and v > 0])
    if vals.size < 3 or np.allclose(vals, vals[0]):
        # Degenerate: map positives to a middle-ish band.
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


# ---------- motion phases & timing (per object) ----------

def classify_step_phase(dx, dy, dz, speed, alt_band, still_thresh):
    horiz = math.sqrt(dx * dx + dz * dz)
    v = abs(dy)

    if speed < still_thresh:
        return f"STILL_{alt_band}"
    if v > horiz and dy > 0:
        return f"ASCEND_{alt_band}"
    if v > horiz and dy < 0:
        return f"DESCEND_{alt_band}"
    if horiz >= v:
        return f"HORIZ_{alt_band}"
    return f"MIXED_{alt_band}"


def extract_motion_phases_and_timeline(frames, px, py, pz, altitude_band_fn,
                                       max_phases=6):
    """
    Returns:
      phases: short list of symbolic phase codes (for MotionPhases)
      timeline: string encoding phases with relative durations
                e.g. "ASCEND_MID:0.20;HORIZ_HIGH:0.50;DESCEND_MID:0.30"
    """
    n = len(frames)
    if n < 2:
        return [], ""

    dx = np.diff(px)
    dy = np.diff(py)
    dz = np.diff(pz)
    dt = np.diff(frames)

    valid = (
        np.isfinite(dx) & np.isfinite(dy) & np.isfinite(dz) &
        np.isfinite(dt) & (dt > 0)
    )
    if not valid.any():
        # no reliable motion; treat as still at median altitude
        med_y = float(np.nanmedian(py)) if np.isfinite(np.nanmedian(py)) else np.nan
        alt = altitude_band_fn(med_y)
        return [f"STILL_{alt}"], f"STILL_{alt}:1.00"

    step_speed = np.zeros(n - 1)
    step_alt = np.empty(n - 1)
    step_alt[:] = np.nan

    for i in range(n - 1):
        if not valid[i]:
            continue
        d = vec_len(dx[i], dy[i], dz[i])
        step_speed[i] = d / dt[i]
        mid_y = 0.5 * (py[i] + py[i + 1])
        step_alt[i] = mid_y

    non_zero_speeds = step_speed[step_speed > 0]
    if non_zero_speeds.size == 0:
        med_y = float(np.nanmedian(py)) if np.isfinite(np.nanmedian(py)) else np.nan
        alt = altitude_band_fn(med_y)
        return [f"STILL_{alt}"], f"STILL_{alt}:1.00"

    speed_scale = np.median(non_zero_speeds)
    still_thresh = 0.1 * speed_scale  # relative to object motion

    # classify each valid step
    step_codes = []
    step_durations = []
    for i in range(n - 1):
        if not valid[i]:
            continue
        s = step_speed[i]
        if not np.isfinite(s):
            continue
        alt_band = altitude_band_fn(step_alt[i])
        code = classify_step_phase(dx[i], dy[i], dz[i], s, alt_band, still_thresh)
        step_codes.append(code)
        step_durations.append(float(dt[i]))

    if not step_codes:
        med_y = float(np.nanmedian(py)) if np.isfinite(np.nanmedian(py)) else np.nan
        alt = altitude_band_fn(med_y)
        return [f"STILL_{alt}"], f"STILL_{alt}:1.00"

    # merge contiguous same-code steps into segments
    segments = []
    cur_code = step_codes[0]
    cur_time = step_durations[0]
    for code, dur in zip(step_codes[1:], step_durations[1:]):
        if code == cur_code:
            cur_time += dur
        else:
            segments.append((cur_code, cur_time))
            cur_code = code
            cur_time = dur
    segments.append((cur_code, cur_time))

    total_time = sum(t for _, t in segments) or 1.0

    # build compact phases (dedup, limited length)
    phase_list = [c for c, _ in segments]
    if len(phase_list) > max_phases:
        idxs = np.linspace(0, len(phase_list) - 1, max_phases).astype(int)
        phase_list = [phase_list[i] for i in idxs]

    # build timeline string
    # (normalize and clip small fractions for readability)
    timeline_parts = []
    for code, t in segments:
        frac = max(0.0, min(1.0, t / total_time))
        if frac < 0.03:
            continue
        timeline_parts.append(f"{code}:{frac:.2f}")
    timeline = ";".join(timeline_parts) if timeline_parts else ""

    return phase_list, timeline


# ---------- OTN-Text: base PathSummary (from v7) ----------

def pathsummary_from_core(stats, phases):
    """
    Build short, deterministic NL summary from:
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


# ---------- Temporal path trace ----------

def build_path_trace(frames, px, py, pz, max_points=12):
    """
    Downsample positions along the trajectory to preserve shape & order.
    Returns a compact string:
      "f0:[x,y,z];f10:[x,y,z];..."
    """
    valid = np.isfinite(px) & np.isfinite(py) & np.isfinite(pz)
    if valid.sum() == 0:
        return ""
    idx = np.where(valid)[0]
    if idx.size <= max_points:
        sel = idx
    else:
        sel = np.linspace(idx[0], idx[-1], max_points).astype(int)
    parts = []
    for i in sel:
        f = int(frames[i])
        x, y, z = float(px[i]), float(py[i]), float(pz[i])
        parts.append(f"f{f}:[{x:.3f},{y:.3f},{z:.3f}]")
    return ";".join(parts)


# ---------- Outlier event summaries ----------

def build_outlier_events(frames, py, step_speeds, dt, valid_steps):
    """
    Returns (max_speed_event_str, largest_vert_change_event_str).
    """
    max_speed_event = ""
    largest_vert_event = ""

    # Max speed event (based on step_speeds over valid steps)
    if valid_steps.any():
        speeds = np.where(valid_steps, step_speeds, 0.0)
        i = int(np.argmax(speeds))
        max_s = float(speeds[i])
        if max_s > 0:
            f0 = int(frames[i])
            f1 = int(frames[i + 1]) if i + 1 < len(frames) else f0
            max_speed_event = f"Peak speed ~{max_s:.3f} between frames {f0}-{f1}."

    # Largest vertical excursion (min->max y)
    finite_y = np.isfinite(py)
    if finite_y.any():
        ys = py.copy()
        ys[~finite_y] = np.nan
        min_y = float(np.nanmin(ys))
        max_y = float(np.nanmax(ys))
        if np.isfinite(min_y) and np.isfinite(max_y) and max_y - min_y > 0:
            min_idx = int(np.nanargmin(ys))
            max_idx = int(np.nanargmax(ys))
            f_min = int(frames[min_idx])
            f_max = int(frames[max_idx])
            if max_y > min_y:
                largest_vert_event = (
                    f"Vertical span from {min_y:.3f} (f{f_min}) "
                    f"to {max_y:.3f} (f{f_max})."
                )

    return max_speed_event, largest_vert_event


# ---------- Semantic role heuristics ----------

def infer_semantic_role(name, category, path_length, vis_ratio):
    """
    Heuristic, scene-neutral semantic roles.
    Uses name/category plus basic motion/visibility.
    """
    nm = (name or "").lower()
    cat = (category or "").lower()
    txt = nm + " " + cat

    def has(*keys):
        return any(k in txt for k in keys)

    moving = path_length > 0.5 and vis_ratio > 0.1

    if has("player", "hero", "pawn", "character"):
        return "Player"
    if has("enemy", "foe", "opponent", "bot", "npc"):
        return "Enemy"
    if has("projectile", "bullet", "missile", "rocket", "arrow"):
        return "Projectile"
    if has("pickup", "collectible", "coin", "item", "loot", "powerup"):
        return "Collectible"
    if has("door", "switch", "lever", "trigger"):
        return "Interactable"
    if has("camera"):
        return "Camera"
    if has("ui", "canvas", "hud"):
        return "UIElement"
    if has("ground", "floor", "terrain", "wall", "platform", "static"):
        return "StaticEnvironment"
    if not moving and vis_ratio > 0.5:
        return "StaticObject"
    if moving and vis_ratio > 0.1:
        return "DynamicObject"
    return "Unknown"


# ---------- Object–object interactions (coarse) ----------

def compute_interaction_summaries(frames_df, object_names,
                                  near_threshold=1.0,
                                  touch_threshold=0.3,
                                  max_partners=3):
    """
    Very lightweight, scene-neutral interaction hints based on minimal distances.

    For each pair of selected objects, we track the minimum distance where both
    have valid positions in the same frame.

    Returns:
      dict: ObjectName -> short interaction summary string.
    """
    names = set(object_names)
    sub = frames_df[frames_df["ObjectName"].isin(names)].copy()

    # Use precomputed numeric positions if available
    if not {"pos_x", "pos_y", "pos_z"}.issubset(sub.columns):
        pos = np.vstack([parse_vec3(v) for v in sub["Position"]])
        sub["pos_x"], sub["pos_y"], sub["pos_z"] = pos[:, 0], pos[:, 1], pos[:, 2]

    # Build per-frame listings
    frame_map = defaultdict(list)
    for _, r in sub.iterrows():
        x, y, z = r["pos_x"], r["pos_y"], r["pos_z"]
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
            continue
        frame = int(r["Frame"])
        obj = str(r["ObjectName"])
        frame_map[frame].append((obj, float(x), float(y), float(z)))

    # Track min distance per unordered pair
    pair_min = defaultdict(lambda: float("inf"))

    for _, objs in frame_map.items():
        n = len(objs)
        for i in range(n):
            oi, x1, y1, z1 = objs[i]
            for j in range(i + 1, n):
                oj, x2, y2, z2 = objs[j]
                if oi == oj:
                    continue
                d = vec_len(x1 - x2, y1 - y2, z1 - z2)
                if d < pair_min[(oi, oj)]:
                    pair_min[(oi, oj)] = d

    # Build per-object textual summaries
    per_obj = defaultdict(list)

    for (oi, oj), d in pair_min.items():
        if not (oi in names and oj in names):
            continue

        # normalize ordering: we recorded as (oi, oj) only once; reflect symmetrically
        def add(obj, other):
            if d <= touch_threshold:
                per_obj[obj].append(f"touches {other}")
            elif d <= near_threshold:
                per_obj[obj].append(f"moves close to {other}")

        add(oi, oj)
        add(oj, oi)

    summaries = {}
    for obj in names:
        events = per_obj.get(obj, [])
        if not events:
            summaries[obj] = ""
            continue
        # deduplicate, cap, make compact
        seen = []
        for e in events:
            if e not in seen:
                seen.append(e)
        seen = seen[:max_partners]
        summaries[obj] = "; ".join(seen) + "."
    return summaries


# ---------- High-level narrative summary ----------

def build_narrative_summary(row):
    """
    Use core stats, bands, phases, interactions and outliers
    to form a compact, chronological-style description.
    """
    name = row.get("ObjectName", "")
    dist_band = row.get("DistBand", "none")
    speed_band = row.get("SpeedBand", "none")
    vert_band = row.get("VertBand", "none")
    motion_phases = (row.get("MotionPhases") or "").split(";") if row.get("MotionPhases") else []
    phase_timeline = row.get("PhaseTimeline", "")
    path_trace = row.get("PathTrace", "")
    interaction = row.get("InteractionSummary", "")
    max_speed_ev = row.get("MaxSpeedEvent", "")
    vert_ev = row.get("LargestVerticalChangeEvent", "")
    role = row.get("SemanticRole", "Unknown")

    parts = []

    # Identity / role
    if role and role != "Unknown":
        parts.append(f"As a {role}, this object")

    # Movement gist from bands
    move_phrase = {
        ("long", "fast"): "travels quickly over a long distance",
        ("long", "medium"): "covers a long distance at a moderate pace",
        ("medium", "fast"): "moves quickly over a medium distance",
        ("medium", "medium"): "moves steadily over a medium distance",
        ("short", "fast"): "moves quickly but only over a short distance",
    }.get((dist_band, speed_band), "")

    if not move_phrase:
        if dist_band == "none" and speed_band == "none":
            move_phrase = "remains mostly in place"
        elif dist_band in ("short", "medium"):
            move_phrase = f"shifts {dist_band} across the scene"
        elif dist_band == "long":
            move_phrase = "moves broadly across the scene"

    if move_phrase:
        if parts:
            parts[-1] += f" {move_phrase}"
        else:
            parts.append(f"It {move_phrase}")

    # Vertical band nuance
    if vert_band == "flat":
        parts.append("with almost no vertical change")
    elif vert_band == "moderate":
        parts.append("with some vertical variation")
    elif vert_band == "large":
        parts.append("with strong changes in height")

    if parts:
        parts[-1] += "."

    # Phase timeline for ordering
    if phase_timeline:
        parts.append(f"Its motion phases over time: {phase_timeline}.")

    # Path trace hint (shape, not all details)
    if path_trace:
        parts.append("Key positions along its path: " + path_trace + ".")

    # Interactions
    if interaction:
        parts.append(f"It {interaction}")

    # Outlier events
    evs = [e for e in [max_speed_ev, vert_ev] if e]
    if evs:
        parts.append("Notable moments: " + " ".join(evs))

    return " ".join(parts).strip()


# ---------- args & main ----------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate trajectories_v8/trajectories.csv with enhanced OTN-Core + OTN-Text."
    )
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

    vol = np.where(
        np.isfinite(rend_vol) & (rend_vol > 0),
        rend_vol,
        np.where(np.isfinite(coll_vol) & (coll_vol > 0), coll_vol, np.nan),
    )
    frames_df["Volume"] = vol

    frames_df = frames_df.sort_values(["ObjectName", "Frame"])

    obj_med_vol = frames_df.groupby("ObjectName")["Volume"].median()
    global_med_vol = float(obj_med_vol[obj_med_vol > 0].median()) if (obj_med_vol > 0).any() else np.nan

    all_y = frames_df["pos_y"].to_numpy()
    altitude_band_fn = make_altitude_band_fn(all_y)

    # ---- First pass: per-object stats + enriched features ----
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

        # Lifecycle: based on presence in frames
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

        # Motion phases & timeline
        phases, phase_timeline = extract_motion_phases_and_timeline(
            frames[idx_valid], px[idx_valid], py[idx_valid], pz[idx_valid],
            altitude_band_fn
        )
        motion_phases_str = ";".join(phases)

        # Temporal path trace
        path_trace = build_path_trace(frames, px, py, pz, max_points=12)

        # Outlier events
        max_speed_event, largest_vert_event = build_outlier_events(
            frames, py, step_speeds, dt, step_valid
        )

        h = hier_info.get(obj, {})
        category = h.get("ObjectCategory", "")

        per_obj.append({
            "ObjectName": obj,
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
            "PhaseTimeline": phase_timeline,
            "PathTrace": path_trace,
            "MaxSpeedEvent": max_speed_event,
            "LargestVerticalChangeEvent": largest_vert_event,
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

    # ---- Second pass: PathSummary, importance, semantic roles ----
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

        semantic_role = infer_semantic_role(
            name=row["ObjectName"],
            category=row["ObjectCategory"],
            path_length=path_length,
            vis_ratio=vis_ratio,
        )

        rec = dict(row)
        rec.update({
            "DistBand": dist_band,
            "SpeedBand": speed_band,
            "VertBand": vert_band,
            "PathSummary": path_summary,
            "SemanticRole": semantic_role,
            "_ImportanceScore": float(importance),
        })
        records.append(rec)

    df = pd.DataFrame(records)

    # ---- Thresholding ----
    move_mask = df["PathLength"] >= args.movement_threshold
    vis_mask = df["VisibilityRatio"] >= args.vis_threshold
    size_mask = df["RelativeSize"].isin(["large", "huge"])
    keep_mask = move_mask | vis_mask | size_mask
    df = df[keep_mask]

    if df.empty:
        print("No trajectories left after applying thresholds.")
        return

    # ---- Sort & top-k ----
    df = df.sort_values("_ImportanceScore", ascending=False)
    if args.top_k and args.top_k > 0 and len(df) > args.top_k:
        df = df.head(args.top_k)

    # ---- Interactions (using final set only) ----
    interaction_map = compute_interaction_summaries(
        frames_df,
        object_names=df["ObjectName"].tolist(),
        near_threshold=1.0,
        touch_threshold=0.3,
        max_partners=3,
    )
    df["InteractionSummary"] = df["ObjectName"].map(interaction_map).fillna("")

    # ---- NarrativeSummary using enriched features ----
    df["NarrativeSummary"] = df.apply(build_narrative_summary, axis=1)

    # ---- Drop internals not intended as final OTN features ----
    df = df.drop(columns=["VertRange", "HorizExtent", "_ImportanceScore"], errors="ignore")

    # ---- Human-readable column names ----
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
        "PhaseTimeline": "PhaseTimeline (Motion phases with approximate relative durations)",
        "PathTrace": "PathTrace (Downsampled key positions along the trajectory)",
        "MaxSpeedEvent": "MaxSpeedEvent (Notable peak speed moment)",
        "LargestVerticalChangeEvent": "LargestVerticalChangeEvent (Notable vertical excursion)",
        "InteractionSummary": "InteractionSummary (Close-approach / touch events with other objects)",
        "SemanticRole": "SemanticRole (Heuristic high-level role: Player/Enemy/Projectile/etc.)",
        "PathSummary": "PathSummary (Short, deterministic natural-language description from core stats)",
        "DistBand": "DistBand (Categorical range of total distance covered: none/short/medium/long)",
        "SpeedBand": "SpeedBand (Categorical range of average movement speed: none/slow/medium/fast)",
        "VertBand": "VertBand (Categorical range of vertical displacement: none/flat/moderate/large)",
        "NarrativeSummary": "NarrativeSummary (Compact narrative description using motion, phases & interactions)",
    }
    df = df.rename(columns=rename_map)

    # ---- Output ----
    outdir = Path(args.outdir) if args.outdir else (Path(args.frames).parent / "trajectories_v8")
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "trajectories.csv"
    df.to_csv(out_path, index=False)

    print(f"✅ Wrote {out_path} with {len(df)} enhanced OTN trajectories (v8).")


if __name__ == "__main__":
    main()

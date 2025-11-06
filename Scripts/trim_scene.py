#!/usr/bin/env python3
"""
trim_scene.py — Reduce Unity scene logs into meaningful, LLM-friendly summaries.

This script filters and summarizes static + frame logs from a Unity game scene.
It keeps only important moving or visible objects and compresses frame data
into keyframes.

Outputs are designed to be interpretable by weaker LLMs for narration or reasoning.

INPUTS:
  --static  path/to/worldGenerated_static.csv
  --frames  path/to/worldGenerated_frames.csv

OUTPUTS:
  objects_summary.csv  — one row per important object (with meaningful metrics)
  keyframes.csv        — compressed movement logs for those objects

USAGE EXAMPLE:
  python trim_scene.py \
    --static "../Data/Slime Rancher/worldGenerated_static.csv" \
    --frames "../Data/Slime Rancher/worldGenerated_frames.csv" \
    --movement-threshold 0.5 \
    --vis-threshold 0.2 \
    --keyframe-epsilon 0.15 \
    --min-frame-gap 2 \
    --top-k 200
"""

import re
import math
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# === Utility regex for number extraction ===
NUM_RE = re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')


# --------------------------------------------------------------------------
# Argument parser
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Trim Unity scene logs for LLM-friendly summarization")
    p.add_argument('--static', required=True, help='path to worldGenerated_static.csv')
    p.add_argument('--frames', required=True, help='path to worldGenerated_frames.csv')
    p.add_argument('--outdir', default=None, help='output directory (default: frames dir /trimmed_scene)')
    p.add_argument('--movement-threshold', type=float, default=0.5, help='min total path length to keep object')
    p.add_argument('--vis-threshold', type=float, default=0.2, help='min fraction of frames with nonzero renderer size')
    p.add_argument('--max-level', type=int, default=None, help='drop objects deeper than this hierarchy Level')
    p.add_argument('--keyframe-epsilon', type=float, default=0.15, help='emit keyframe if moved >= this since last')
    p.add_argument('--min-frame-gap', type=int, default=2, help='emit keyframe if this many frames elapsed')
    p.add_argument('--top-k', type=int, default=200, help='keep only the top-K most important objects')
    return p.parse_args()


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------
def extract_xyz(series: pd.Series):
    """Extract first three numeric values (x, y, z) from stringified tuples or arrays."""
    s = series.astype(str).str.replace('[()\\[\\]]', ' ', regex=True)
    def to3(x):
        nums = NUM_RE.findall(x)
        if len(nums) >= 3:
            return float(nums[0]), float(nums[1]), float(nums[2])
        return np.nan, np.nan, np.nan
    arr = np.vstack(series.fillna('').map(to3).values)
    return pd.DataFrame(arr, columns=['x', 'y', 'z'], index=series.index)


def product_xyz(df_xyz: pd.DataFrame):
    """Compute absolute product of x, y, z (proxy for 3D volume or screen size)."""
    a = df_xyz[['x','y','z']].astype(float)
    a = a.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return (a['x'].abs() * a['y'].abs() * a['z'].abs())


def group_path_length(df_sorted):
    """Compute total path length by summing Euclidean diffs per object."""
    dx = df_sorted['x'].diff()
    dy = df_sorted['y'].diff()
    dz = df_sorted['z'].diff()
    boundary = df_sorted['ObjectName'] != df_sorted['ObjectName'].shift(1)
    dx[boundary] = 0.0
    dy[boundary] = 0.0
    dz[boundary] = 0.0
    step = np.sqrt(dx*dx + dy*dy + dz*dz)
    return step


# --------------------------------------------------------------------------
# Main processing
# --------------------------------------------------------------------------
def main():
    args = parse_args()
    static_df = pd.read_csv(args.static)
    frames_df = pd.read_csv(args.frames)

    # Normalize column names
    frames_df = frames_df.rename(columns={
        'ObjectName': 'ObjectName',
        'Frame': 'Frame',
        'Position': 'Position',
        'RotationEuler': 'RotationEuler',
        'RendererSize': 'RendererSize',
        'ColliderSize': 'ColliderSize'
    })

    # Merge minimal static info for type, hierarchy, etc.
    static_cols = ['ObjectName', 'Level', 'Type', 'Active', 'InView', 'HasRenderer', 'HasCollider']
    static_slim = static_df[static_cols].drop_duplicates(subset=['ObjectName'], keep='first')
    frames_df = frames_df.merge(static_slim, how='left', on='ObjectName')

    # Extract 3D numeric values
    pos_xyz = extract_xyz(frames_df['Position'])
    frames_df = pd.concat([frames_df, pos_xyz], axis=1)
    size_xyz = extract_xyz(frames_df['RendererSize'])
    frames_df['RendererProd'] = product_xyz(size_xyz)

    frames_df = frames_df.sort_values(['ObjectName','Frame'])

    # Movement metrics
    frames_df['StepLen'] = group_path_length(frames_df)

    # Aggregate per object
    agg = frames_df.groupby('ObjectName').agg(
        FrameCount=('Frame','count'),
        FirstFrame=('Frame','min'),
        LastFrame=('Frame','max'),
        TotalPathLength=('StepLen','sum'),
        MaxStepDistance=('StepLen','max'),
        StartX=('x','first'), StartY=('y','first'), StartZ=('z','first'),
        EndX=('x','last'), EndY=('y','last'), EndZ=('z','last'),
        VisibleFrames=('RendererProd', lambda s: (s>0).sum()),
        MedianRenderVolume=('RendererProd','median'),
        HierarchyLevel=('Level','first'),
        ObjectType=('Type','first'),
        IsActive=('Active','first'),
        HasRenderer=('HasRenderer','first'),
        HasCollider=('HasCollider','first')
    ).reset_index()

    # Derived metrics
    agg['NetDisplacement'] = np.sqrt((agg['EndX']-agg['StartX'])**2 +
                                     (agg['EndY']-agg['StartY'])**2 +
                                     (agg['EndZ']-agg['StartZ'])**2)
    agg['VisibilityRatio'] = agg['VisibleFrames'] / agg['FrameCount'].clip(lower=1)
    agg['ImportanceScore'] = (
        agg['TotalPathLength'] *
        (1.0 + agg['MedianRenderVolume']).pow(0.25) *
        (0.5 + agg['VisibilityRatio'])
    )

    # Filter based on movement, visibility, and level
    mask = (agg['TotalPathLength'] >= args.movement_threshold) & (agg['VisibilityRatio'] >= args.vis_threshold)
    if args.max_level is not None:
        mask &= (agg['HierarchyLevel'].fillna(9999) <= args.max_level)
    trimmed = agg[mask].copy().sort_values(['ImportanceScore','TotalPathLength','MedianRenderVolume'], ascending=False)
    if args.top_k and args.top_k > 0:
        trimmed = trimmed.head(args.top_k)

    # Generate keyframes for retained objects
    kept = set(trimmed['ObjectName'])
    f = frames_df[frames_df['ObjectName'].isin(kept)].copy()
    f['Emit'] = False

    last_pos = {}
    eps = args.keyframe_epsilon
    for idx, row in f.iterrows():
        name, frame = row['ObjectName'], int(row['Frame'])
        x, y, z = float(row['x']), float(row['y']), float(row['z'])
        if name not in last_pos:
            f.at[idx, 'Emit'] = True
            last_pos[name] = (x, y, z, frame)
        else:
            lx, ly, lz, lf = last_pos[name]
            moved = math.sqrt((x-lx)**2 + (y-ly)**2 + (z-lz)**2)
            gap = frame - lf
            if moved >= eps or gap >= args.min_frame_gap:
                f.at[idx, 'Emit'] = True
                last_pos[name] = (x, y, z, frame)

    keyframes = f[f['Emit']][['Frame','ObjectName','Position','RotationEuler','RendererSize','ColliderSize']].copy()

    # ----------------------------------------------------------------------
    # Rename columns for LLM-friendly schema
    # ----------------------------------------------------------------------
    trimmed = trimmed.rename(columns={
        "FrameCount": "FrameCount (frames recorded)",
        "FirstFrame": "FirstFrame (start frame)",
        "LastFrame": "LastFrame (end frame)",
        "TotalPathLength": "TotalPathLength (total movement distance)",
        "MaxStepDistance": "MaxStepDistance (largest single movement)",
        "NetDisplacement": "NetDisplacement (start to end distance)",
        "VisibilityRatio": "VisibilityRatio (fraction visible)",
        "MedianRenderVolume": "MedianRenderVolume (typical on-screen volume)",
        "HierarchyLevel": "HierarchyLevel (depth in scene tree)",
        "ObjectType": "ObjectType (Unity class/type)",
        "IsActive": "IsActive (true/false)",
        "ImportanceScore": "ImportanceScore (movement × visibility × size)"
    })

    keyframes = keyframes.rename(columns={
        "Frame": "FrameIndex",
        "Position": "WorldPosition (x,y,z)"
    })

    # ----------------------------------------------------------------------
    # Save outputs
    # ----------------------------------------------------------------------
    outdir = Path(args.outdir) if args.outdir else (Path(args.frames).parent / 'trimmed_scene')
    outdir.mkdir(parents=True, exist_ok=True)
    summary_out = outdir / 'objects_summary.csv'
    keyframes_out = outdir / 'keyframes.csv'

    trimmed.to_csv(summary_out, index=False)
    keyframes.to_csv(keyframes_out, index=False)

    print(f"✅ Wrote {summary_out} ({len(trimmed)} objects)")
    print(f"✅ Wrote {keyframes_out} ({len(keyframes)} keyframes)")


if __name__ == "__main__":
    main()

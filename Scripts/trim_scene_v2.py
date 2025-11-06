#!/usr/bin/env python3
"""
trim_scene_v2.py — LLM-friendly Unity scene log compressor (with frame skipping)

This version adds:
  • frame subsampling (--frame-step)
  • adaptive keyframe selection (based on movement speed)
  • same human-readable columns as v1

Outputs:
  objects_summary.csv — per-object summary
  keyframes.csv — compressed movement sequence

Recommended usage:
  python trim_scene_v2.py \
    --static "../Data/Slime Rancher/worldGenerated_static.csv" \
    --frames "../Data/Slime Rancher/worldGenerated_frames.csv" \
    --movement-threshold 0.5 \
    --vis-threshold 0.2 \
    --keyframe-epsilon 0.15 \
    --min-frame-gap 2 \
    --frame-step 2 \
    --top-k 200
"""

import re
import math
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

NUM_RE = re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')


# ---------------------- Utility Functions ----------------------
def extract_xyz(series: pd.Series):
    """Extract first 3 numeric values from a string representing (x,y,z)."""
    s = series.astype(str).str.replace('[()\\[\\]]', ' ', regex=True)
    def to3(x):
        nums = NUM_RE.findall(x)
        if len(nums) >= 3:
            return float(nums[0]), float(nums[1]), float(nums[2])
        return np.nan, np.nan, np.nan
    arr = np.vstack(series.fillna('').map(to3).values)
    return pd.DataFrame(arr, columns=['x','y','z'], index=series.index)


def product_xyz(df_xyz: pd.DataFrame):
    """Compute product of |x*y*z| as a proxy for visible volume."""
    a = df_xyz[['x','y','z']].astype(float)
    a = a.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return (a['x'].abs() * a['y'].abs() * a['z'].abs())


def group_path_length(df_sorted):
    """Compute distance moved per frame per object."""
    dx = df_sorted['x'].diff()
    dy = df_sorted['y'].diff()
    dz = df_sorted['z'].diff()
    boundary = df_sorted['ObjectName'] != df_sorted['ObjectName'].shift(1)
    dx[boundary] = dy[boundary] = dz[boundary] = 0.0
    return np.sqrt(dx*dx + dy*dy + dz*dz)


# ---------------------- Argument Parser ----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Trim Unity scene logs (Version 2 with frame skipping)")
    p.add_argument('--static', required=True)
    p.add_argument('--frames', required=True)
    p.add_argument('--outdir', default=None)
    p.add_argument('--movement-threshold', type=float, default=0.5)
    p.add_argument('--vis-threshold', type=float, default=0.2)
    p.add_argument('--max-level', type=int, default=None)
    p.add_argument('--keyframe-epsilon', type=float, default=0.15)
    p.add_argument('--min-frame-gap', type=int, default=2)
    p.add_argument('--frame-step', type=int, default=2, help='keep every Nth frame globally (default=2)')
    p.add_argument('--top-k', type=int, default=200)
    return p.parse_args()


# ---------------------- Main ----------------------
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

    # Merge slim static info
    static_cols = ['ObjectName','Level','Type','Active','InView','HasRenderer','HasCollider']
    static_slim = static_df[static_cols].drop_duplicates(subset=['ObjectName'], keep='first')
    frames_df = frames_df.merge(static_slim, how='left', on='ObjectName')

    # Extract numeric values
    pos_xyz = extract_xyz(frames_df['Position'])
    frames_df = pd.concat([frames_df, pos_xyz], axis=1)
    size_xyz = extract_xyz(frames_df['RendererSize'])
    frames_df['RendererProd'] = product_xyz(size_xyz)

    # Sort
    frames_df = frames_df.sort_values(['ObjectName','Frame'])

    # Subsample every Nth frame but ALWAYS keep first/last frame per object
    if args.frame_step > 1:
        first_per_obj = frames_df.groupby('ObjectName')['Frame'].transform('min')
        last_per_obj  = frames_df.groupby('ObjectName')['Frame'].transform('max')

        # Keep: first frame for the object, last frame for the object,
        # and every Nth frame offset from that object's first frame
        keep = (
            (frames_df['Frame'] == first_per_obj) |
            (frames_df['Frame'] == last_per_obj)  |
            (((frames_df['Frame'] - first_per_obj) % args.frame_step) == 0)
        )
        frames_df = frames_df[keep]


    # Compute per-frame step distance
    frames_df['StepLen'] = group_path_length(frames_df)

    # Aggregate
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

    # Filter low-importance
    mask = (agg['TotalPathLength'] >= args.movement_threshold) & (agg['VisibilityRatio'] >= args.vis_threshold)
    if args.max_level is not None:
        mask &= (agg['HierarchyLevel'].fillna(9999) <= args.max_level)
    trimmed = agg[mask].sort_values(['ImportanceScore','TotalPathLength','MedianRenderVolume'], ascending=False)
    if args.top_k and args.top_k > 0:
        trimmed = trimmed.head(args.top_k)

    # Adaptive keyframe compression
    kept = set(trimmed['ObjectName'])
    f = frames_df[frames_df['ObjectName'].isin(kept)].copy()
    f['Emit'] = False

    last_pos = {}
    eps = args.keyframe_epsilon
    for idx, row in f.iterrows():
        name, frame = row['ObjectName'], int(row['Frame'])
        x, y, z = float(row['x']), float(row['y']), float(row['z'])
        if name not in last_pos:
            f.at[idx,'Emit'] = True
            last_pos[name] = (x,y,z,frame)
        else:
            lx,ly,lz,lf = last_pos[name]
            moved = math.sqrt((x-lx)**2 + (y-ly)**2 + (z-lz)**2)
            gap = frame - lf
            # Adaptive: skip small movement for slow objects
            if moved >= eps or gap >= args.min_frame_gap or frame == f['Frame'].max():
                f.at[idx,'Emit'] = True
                last_pos[name] = (x,y,z,frame)

    keyframes = f[f['Emit']][['Frame','ObjectName','Position','RotationEuler','RendererSize','ColliderSize']]

    # Rename for clarity
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

    # Save
    outdir = Path(args.outdir) if args.outdir else (Path(args.frames).parent / 'trimmed_scene_v2')
    outdir.mkdir(parents=True, exist_ok=True)
    summary_out = outdir / 'objects_summary.csv'
    keyframes_out = outdir / 'keyframes.csv'

    trimmed.to_csv(summary_out, index=False)
    keyframes.to_csv(keyframes_out, index=False)

    print(f"✅ Wrote {summary_out} ({len(trimmed)} objects)")
    print(f"✅ Wrote {keyframes_out} ({len(keyframes)} keyframes after skipping frames)")


if __name__ == "__main__":
    main()

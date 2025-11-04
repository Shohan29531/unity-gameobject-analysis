import pandas as pd
import numpy as np
import re
from math import sqrt

def parse_tuple(s):
    """Convert '(x, y, z)' → (x, y, z)"""
    if isinstance(s, str) and re.match(r"^\(.*\)$", s.strip()):
        try:
            return tuple(map(float, s.strip("() ").split(",")))
        except Exception:
            return None
    return None

def object_has_motion(positions, threshold=1e-3):
    """Return True if object moved more than threshold distance."""
    positions = [p for p in positions if p is not None]
    if len(positions) < 2:
        return False
    max_dist = 0
    for i in range(len(positions)-1):
        for j in range(i+1, len(positions)):
            x1, y1, z1 = positions[i]
            x2, y2, z2 = positions[j]
            d = sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            max_dist = max(max_dist, d)
    return max_dist >= threshold

def clean_object_positions(input_csv="object_positions.csv",
                           output_csv="object_positions_cleaned.csv",
                           motion_threshold=1e-3,
                           drop_static=True,
                           drop_zeros=True,
                           min_valid_frames=2):
    # ✅ Safe reading: handle commas inside quotes
    df = pd.read_csv(
        input_csv,
        engine="python",          # handles complex quoting better
        quotechar='"',
        skip_blank_lines=True,
        dtype=str,
        on_bad_lines='skip'       # skip malformed rows
    )

    # Auto-detect frame columns
    frame_cols = [c for c in df.columns if str(c).startswith("Frame_")]
    if not frame_cols:
        print("⚠️ No Frame_* columns found — check CSV formatting.")
        return

    keep_rows = []
    for i, row in df.iterrows():
        positions = [parse_tuple(row[c]) for c in frame_cols]
        non_none = [p for p in positions if p is not None]
        if len(non_none) < min_valid_frames:
            continue
        if drop_zeros and all(abs(x)<1e-6 and abs(y)<1e-6 and abs(z)<1e-6 for (x,y,z) in non_none):
            continue
        if drop_static and not object_has_motion(non_none, threshold=motion_threshold):
            continue
        keep_rows.append(i)

    cleaned = df.loc[keep_rows]
    cleaned.to_csv(output_csv, index=False)
    print(f"✅ Cleaned file written to {output_csv} — kept {len(cleaned)} of {len(df)} rows.")

if __name__ == "__main__":
    clean_object_positions()

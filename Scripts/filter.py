import pandas as pd
from pathlib import Path

def analyze_static_csv(input_file="../Data/Slime Rancher/worldGenerated_static.csv"):
    """
    Analyze how many objects have colliders and/or renderers,
    focusing on those that are InView=True and Active=True.
    Also reports how many are visible + interactive (have collider or renderer).
    """
    # --- Validate path ---
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"❌ File not found: {input_path.resolve()}")

    print(f"📂 Reading file: {input_path.resolve()}")
    df = pd.read_csv(input_path)
    df.columns = [c.strip() for c in df.columns]

    # --- Normalize boolean columns ---
    for col in ["HasCollider", "HasRenderer", "InView", "Active"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"])
        else:
            raise KeyError(f"Missing column: {col}")

    # --- Filter visible & active ---
    visible_active = df[(df["InView"]) & (df["Active"])]

    print(f"\n🔍 Considering only objects that are InView=True and Active=True")
    print(f"Filtered objects: {len(visible_active)} / {len(df)} total")

    # --- Logical groups ---
    both = visible_active[visible_active["HasCollider"] & visible_active["HasRenderer"]]
    either = visible_active[visible_active["HasCollider"] | visible_active["HasRenderer"]]
    only_collider = visible_active[visible_active["HasCollider"] & ~visible_active["HasRenderer"]]
    only_renderer = visible_active[visible_active["HasRenderer"] & ~visible_active["HasCollider"]]
    neither = visible_active[~(visible_active["HasCollider"] | visible_active["HasRenderer"])]

    # --- New group: has collider OR renderer, AND in-view AND active ---
    visible_and_interactive = df[
        (df["InView"]) & (df["Active"]) & (df["HasCollider"] | df["HasRenderer"])
    ]

    # --- Print summary ---
    print("\n📊 Collider / Renderer Summary (Visible & Active only)")
    print("------------------------------------------------------")
    print(f"Visible + Active objects     : {len(visible_active)}")
    print(f"Has BOTH collider & renderer : {len(both)}")
    print(f"Has EITHER collider/renderer : {len(either)}")
    print(f"Has ONLY collider            : {len(only_collider)}")
    print(f"Has ONLY renderer            : {len(only_renderer)}")
    print(f"Has NEITHER                  : {len(neither)}")

    print("\n✨ Visible + Interactive (InView & Active & HasCollider/Renderer):")
    print(f"Count : {len(visible_and_interactive)}")

    # --- Optional: save that subset ---
    visible_and_interactive.to_csv(input_path.parent / "visible_and_interactive.csv", index=False)

    return {
        "total": len(df),
        "visible_active": len(visible_active),
        "both": len(both),
        "either": len(either),
        "only_collider": len(only_collider),
        "only_renderer": len(only_renderer),
        "neither": len(neither),
        "visible_and_interactive": len(visible_and_interactive)
    }


if __name__ == "__main__":
    stats = analyze_static_csv()
    # print("\nSummary dict:", stats)

import pandas as pd

# === CONFIGURATION SECTION ===
# Path to your static CSV file
# csv_path = "../Data/Slime Rancher/worldGenerated_static.csv"
csv_path = "../Data/Heli/HelicopterPhysics_static.csv"

# --- Filter Settings ---
# You can flip these True/False or set thresholds
filter_active = True          # only include objects that are active
filter_inview = True          # only include objects currently in camera view
filter_has_collider = True    # only include objects that have a collider
filter_has_renderer = True    # only include objects that have a renderer
max_level = 1                 # only include objects with Level >= this number

# === END CONFIGURATION ===


def load_and_filter_static_log(path):
    # Read CSV (auto-detects commas, handles quoted values)
    df = pd.read_csv(path)

    # Ensure boolean-like columns are treated as booleans
    for col in ["Active", "InView", "HasCollider", "HasRenderer"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False})

    # Apply filters step by step
    mask = pd.Series(True, index=df.index)

    # if "Active" in df.columns:
    #     mask &= (df["Active"] == filter_active)
    # if "InView" in df.columns:
    #     mask &= (df["InView"] == filter_inview)
    # if "HasCollider" in df.columns:
    #     mask &= (df["HasCollider"] == filter_has_collider)
    # if "HasRenderer" in df.columns:
    #     mask &= (df["HasRenderer"] == filter_has_renderer)
    if "Level" in df.columns:
        mask &= (df["Level"] <=  max_level)

    filtered_df = df[mask].reset_index(drop=True)
    return filtered_df


if __name__ == "__main__":
    filtered = load_and_filter_static_log(csv_path)
    print(f"\n✅ {len(filtered)} objects matched the filters.\n")

    # Print only object names
    if "ObjectName" in filtered.columns:
        for name in filtered["ObjectName"]:
            print(name)
    else:
        print("⚠️ No 'ObjectName' column found in the CSV.")


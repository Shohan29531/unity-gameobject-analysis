import pandas as pd
import plotly.express as px
import os

# === CONFIGURATION ===
csv_path = "../Data/Heli/HelicopterPhysics_static.csv"

# --- Load & Filter ---
df = pd.read_csv(csv_path)

# Normalize boolean columns
for col in ["HasRenderer", "HasCollider"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False})

# Keep only rows with renderer or collider
df = df[(df["HasRenderer"] == True) | (df["HasCollider"] == True)]

# --- Build hierarchy from ObjectName ---
paths = df["ObjectName"].dropna().tolist()

rows = []
for path in paths:
    parts = [p for p in path.split("/") if p.strip()]
    for i in range(len(parts)):
        full_path = "/".join(parts[: i + 1])
        parent = "/".join(parts[:i]) if i > 0 else ""
        rows.append({"id": full_path, "parent": parent})

hier_df = pd.DataFrame(rows).drop_duplicates(subset=["id"])
# Merge with type / metadata for coloring
hier_df = hier_df.merge(df[["ObjectName", "Type"]], how="left",
                        left_on="id", right_on="ObjectName")

# Replace missing types (intermediate nodes)
hier_df["Type"] = hier_df["Type"].fillna("Group")

# --- Plotly Treemap ---
fig = px.treemap(
    hier_df,
    names="id",
    parents="parent",
    color="Type",
)

fig.update_traces(root_color="lightgrey")
fig.update_layout(
    title="Helicopter Object Hierarchy (Renderer/Collider Only)",
    margin=dict(t=50, l=25, r=25, b=25),
)

fig.show()

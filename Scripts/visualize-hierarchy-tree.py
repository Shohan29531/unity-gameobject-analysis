import pandas as pd
from pyvis.network import Network
import os

# === CONFIGURATION ===
csv_path = "../Data/Heli/HelicopterPhysics_static.csv"
output_html = "HelicopterHierarchy.html"

# --- Load & Filter ---
df = pd.read_csv(csv_path)

# Normalize booleans
for col in ["HasRenderer", "HasCollider"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False})

# Keep only relevant objects
df = df[(df["HasRenderer"] == True) | (df["HasCollider"] == True)]

# --- Build network graph ---
net = Network(height="850px", width="100%", directed=True, notebook=False, bgcolor="#0d1117", font_color="white")
net.toggle_physics(False)  # keep tree layout stable

for _, row in df.iterrows():
    path = row["ObjectName"]
    parts = [p for p in path.split("/") if p.strip()]
    # Create nodes and edges for each part
    for i in range(len(parts)):
        node = "/".join(parts[:i+1])
        parent = "/".join(parts[:i]) if i > 0 else None

        # Node color: blue = renderer, red = collider, purple = both, gray = group
        has_rend = bool(row.get("HasRenderer", False))
        has_coll = bool(row.get("HasCollider", False))
        if has_rend and has_coll:
            color = "#a066ff"  # purple
        elif has_rend:
            color = "#00b4d8"  # blue
        elif has_coll:
            color = "#ff4d6d"  # red
        else:
            color = "#999999"  # gray

        net.add_node(node, label=parts[i], color=color, title=node)
        if parent:
            net.add_edge(parent, node)

# --- Layout configuration ---
net.set_options("""
var options = {
  "layout": {
    "hierarchical": {
      "enabled": true,
      "direction": "UD",
      "sortMethod": "directed"
    }
  },
  "physics": {
    "hierarchicalRepulsion": {
      "nodeDistance": 150
    },
    "solver": "hierarchicalRepulsion"
  },
  "interaction": {
    "dragNodes": true,
    "dragView": true,
    "zoomView": true
  }
}
""")

# --- Save to HTML safely ---
net.write_html(output_html, open_browser=False)
print(f"✅ Interactive hierarchy saved to: {os.path.abspath(output_html)}")
print("💡 Open it in your browser to explore (drag, zoom, fold branches).")

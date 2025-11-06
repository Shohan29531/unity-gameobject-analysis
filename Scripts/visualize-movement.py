import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import ast

# === CONFIGURATION ===
# csv_path = "../Data/Slime Rancher/worldGenerated_frames.csv"
csv_path = "../Data/Heli/HelicopterPhysics_frames.csv"
# csv_path = "../Data/Beat Saber/StandardGameplay_frames.csv"

# Filter a single object (use None for all)
object_filter = None  # e.g. "Rotor"

# --- NEW FILTER ---
max_level = 10  # exclude objects with Level > this number

# --- Helper ---
def parse_triplet(triplet_str):
    try:
        val = ast.literal_eval(triplet_str)
        if isinstance(val, (list, tuple)) and len(val) >= 3:
            return [float(val[0]), float(val[1]), float(val[2])]
    except Exception:
        pass
    return [None, None, None]

# --- Load data ---
df = pd.read_csv(csv_path)

# Parse position
df[["x", "y", "z"]] = df["Position"].apply(parse_triplet).apply(pd.Series)

# Parse renderer and collider centers
df[["rx", "ry", "rz"]] = df["RendererCenter"].apply(parse_triplet).apply(pd.Series)
df[["cx", "cy", "cz"]] = df["ColliderCenter"].apply(parse_triplet).apply(pd.Series)

# Keep only rows where at least one renderer/collider coordinate exists
has_renderer = df[["rx", "ry", "rz"]].notna().any(axis=1)
has_collider = df[["cx", "cy", "cz"]].notna().any(axis=1)
df = df[has_renderer | has_collider]

# --- INFER LEVEL FROM NAME ---
df["Level"] = df["ObjectName"].astype(str).apply(lambda name: name.count("/"))

# --- APPLY LEVEL FILTER ---
df = df[df["Level"] <= max_level]

# Drop rows with no position data (shouldn't happen, but safe)
df = df.dropna(subset=["x", "y", "z"])

# Optional: filter a single object name
if object_filter:
    df = df[df["ObjectName"].str.contains(object_filter, case=False)]

# # --- EXCLUDE STATIC OBJECTS (never move across frames) ---
# # Group by object and check if position ever changes
# moving_objects = (
#     df.groupby("ObjectName")[["x", "y", "z"]]
#     .agg(lambda col: col.nunique() > 1)
#     .any(axis=1)
# )
# moving_objects = moving_objects[moving_objects].index  # only keep True ones
# df = df[df["ObjectName"].isin(moving_objects)]

# --- SAFETY CHECK: Ensure we have data after filtering ---
if df.empty:
    print("⚠️ No objects passed the filters (check max_level or movement). Exiting.")
    exit()

objects = sorted(df["ObjectName"].unique())
frames = sorted(df["Frame"].unique())

print(f"✅ Loaded {len(df)} records for {len(objects)} moving objects across {len(frames)} frames.")
print(f"ℹ️ Showing only moving objects with renderer/collider data and Level ≤ {max_level}.")

# --- 3D Plot setup ---
fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(bottom=0.15)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Filtered Object Movement Viewer")

colors = plt.cm.tab10.colors

# Create a point and a trail per object
points, trails = {}, {}
for i, obj in enumerate(objects):
    (pnt,) = ax.plot([], [], [], "o", color=colors[i % len(colors)], label=obj)
    (trl,) = ax.plot([], [], [], "-", color=colors[i % len(colors)], alpha=0.4)
    points[obj] = pnt
    trails[obj] = trl

# --- SAFETY CHECK: Avoid NaN axis limits ---
if df[["x", "y", "z"]].dropna().empty:
    print("⚠️ All positions are NaN — cannot set axis limits.")
    exit()

# Move legend outside the 3D plot area 
if len(objects) > 0:
    ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplots_adjust(right=0.8)  # shrink plot area so legend fits nicely

# Set axis limits
ax.set_xlim(df["x"].min(), df["x"].max())
ax.set_ylim(df["y"].min(), df["y"].max())
ax.set_zlim(df["z"].min(), df["z"].max())

# --- Slider ---
ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
slider = Slider(ax_slider, "Frame", frames[0], frames[-1],
                valinit=frames[0], valstep=1)

# --- Update function ---
def update(val):
    frame = int(slider.val)
    frame_data = df[df["Frame"] == frame]
    ax.set_title(f"Frame {frame}")

    for obj in objects:
        obj_data = frame_data[frame_data["ObjectName"] == obj]
        if len(obj_data) == 0:
            continue
        x, y, z = obj_data[["x", "y", "z"]].values[0]
        points[obj].set_data([x], [y])
        points[obj].set_3d_properties([z])

        trail_data = df[(df["ObjectName"] == obj) & (df["Frame"] <= frame)]
        trails[obj].set_data(trail_data["x"], trail_data["y"])
        trails[obj].set_3d_properties(trail_data["z"])

    fig.canvas.draw_idle()

slider.on_changed(update)
update(frames[0])

plt.show()

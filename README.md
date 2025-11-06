# 🎮 Unity Scene Trimmer & Summarizer

A lightweight Python toolkit for compressing **Unity game scene logs** into compact, **LLM-friendly summaries**.  
It filters, deduplicates, and samples high-frequency frame data — keeping only the most meaningful movements and visual changes.  
Perfect for downstream scene narration or analysis by smaller language models.

---

## ✨ Features

- 🧱 **Static + Frame log fusion**  
  Combines `worldGenerated_static.csv` and `worldGenerated_frames.csv` to produce scene summaries.

- 🧭 **Smart trimming**  
  Removes invisible or static objects using configurable movement and visibility thresholds.

- 🎞️ **Keyframe compression**  
  Keeps only significant motion changes with adaptive per-object keyframe detection.

- ⏱️ **Frame skipping (v2)**  
  Optionally record every _N_th frame to reduce file size without losing meaningful dynamics.

- 🧩 **LLM-readable output**  
  Produces human-descriptive column names (e.g. `TotalPathLength (total movement distance)`)  
  to help smaller models describe scenes in plain English.

---

## 🧠 How It Works

1. **`trim_scene.py`**  
   - Filters out low-motion and low-visibility objects.  
   - Summarizes object stats (movement, visibility, size, etc.).  
   - Generates `objects_summary.csv` and `keyframes.csv`.

2. **`trim_scene_v2.py`**  
   - Adds frame skipping and adaptive keyframe selection.  
   - Always keeps first/last frames for continuity.  
   - Outputs to `trimmed_scene_v2/`.

Both scripts produce data concise enough for LLM-based scene description tasks.

---

## 📂 Output Files

### `objects_summary.csv`
Summarized information per object:
| Column | Meaning |
|---------|----------|
| `ObjectName` | Full Unity hierarchy path (e.g. `A/B/C`) |
| `FrameCount (frames recorded)` | Number of frames where object appeared |
| `FirstFrame (start frame)` | First recorded frame |
| `LastFrame (end frame)` | Last recorded frame |
| `TotalPathLength (total movement distance)` | 3D distance traveled |
| `VisibilityRatio (fraction visible)` | Portion of frames object was visible |
| `MedianRenderVolume (typical on-screen volume)` | Proxy for size/importance |
| `ImportanceScore (movement × visibility × size)` | Ranking metric |

---

### `keyframes.csv`
Condensed motion timeline for important objects:
| Column | Meaning |
|---------|----------|
| `FrameIndex` | Frame number |
| `ObjectName` | GameObject path |
| `WorldPosition (x,y,z)` | Object position |
| `RotationEuler (x,y,z)` | Object rotation |
| `RendererSize (w,h,d)` | Visible size proxy |
| `ColliderSize (w,h,d)` | Collider size proxy |

---

## ⚙️ Usage

```python trim_scene.py \
  --static "../Data/Slime Rancher/worldGenerated_static.csv" \
  --frames "../Data/Slime Rancher/worldGenerated_frames.csv" \
  --movement-threshold 0.5 \
  --vis-threshold 0.2 \
  --keyframe-epsilon 0.15 \
  --min-frame-gap 2 \
  --top-k 200'''


or the optimized version (recommended):

'''python trim_scene_v2.py \
  --static "../Data/Slime Rancher/worldGenerated_static.csv" \
  --frames "../Data/Slime Rancher/worldGenerated_frames.csv" \
  --movement-threshold 0.5 \
  --vis-threshold 0.2 \
  --keyframe-epsilon 0.15 \
  --min-frame-gap 2 \
  --frame-step 2 \
  --top-k 200'''

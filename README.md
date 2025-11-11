# Unity Scene Trimmer & Narrative Summarizer

A single Python script that converts high-frequency Unity telemetry into a compact, **LLM-friendly, object-centric narrative table**.  
It preserves **ordering, interactions, and distinctive motion patterns** while drastically reducing file size.

- **Inputs:** `worldGenerated_static.csv` + `worldGenerated_frames.csv`  
- **Output:** `trajectories/trajectories.csv` (one row per salient object)

---

## What It Produces

**OTN-Core (structured stats)**  
- Identity & lifecycle (names, frames, positions)  
- Motion & visibility (distance, displacement, speed, visibility)  
- Quantile-based **bands** (distance / speed / vertical)

**Temporal & shape cues**  
- **PathTrace:** downsampled `(frame → [x,y,z])` waypoints to keep path shape & order  
- **PhaseTimeline:** motion phases (e.g., `ASCEND_MID`) with **relative durations**

**Symbolic motion & semantics**  
- **MotionPhases:** deduplicated sequence of symbolic motion labels  
- **SemanticRole:** heuristic role (e.g., `Player`, `Enemy`, `Projectile`, `StaticEnvironment`)

**Interactions & notable events**  
- **InteractionSummary:** coarse “moves close to / touches” other objects  
- **MaxSpeedEvent / LargestVerticalChangeEvent:** highlight peak moments

**Narrative layers**  
- **PathSummary:** deterministic text from core stats  
- **NarrativeSummary:** compact, chronological description that blends phases, traces, and interactions

---

## Columns Overview

### Identity & Lifecycle
| Column | Meaning |
|---|---|
| `ObjectName (Name of the Object)` | The Unity object’s name |
| `ObjectCategory (Potential Object Type)` | Derived from Type/Tag/Layer |
| `StartFrame (Frame where the object first appears)` | First seen frame |
| `EndFrame (Frame where the object last appears)` | Last seen frame |

### Motion & Visibility
| Column | Meaning |
|---|---|
| `StartPosition [x,y,z] at first valid position` | First valid world position |
| `EndPosition [x,y,z] at last valid position` | Last valid world position |
| `PathLength (Sum of all displacements, added stepwise with abs value)` | Total path distance |
| `NetDisplacement (Vector value of end position - start position)` | Start→End straight-line distance |
| `AvgSpeed (Mean step speed over valid steps)` | Average speed over valid intervals |
| `VisibilityRatio (Percentage of frames where the object is visible)` | Fraction visible |
| `SizeChange (Change in perceived size of the object, can be due to camera movement)` | `grow/shrink/stable/unknown` |
| `RelativeSize (Compared to other objects in the game)` | `tiny/small/medium/large/huge/unknown` |

### Normalized Bands (scene-relative)
| Column | Meaning |
|---|---|
| `DistBand (Categorical range of total distance covered: none/short/medium/long)` | Quantile-based distance category |
| `SpeedBand (Categorical range of average movement speed: none/slow/medium/fast)` | Quantile-based speed category |
| `VertBand (Categorical range of vertical displacement: none/flat/moderate/large)` | Quantile-based vertical category |

### Temporal & Shape Cues
| Column | Meaning |
|---|---|
| `PathTrace (Downsampled key positions along the trajectory)` | e.g., `f0:[x,y,z];f10:[x,y,z];…` |
| `PhaseTimeline (Motion phases with approximate relative durations)` | e.g., `ASCEND_MID:0.20;HORIZ_HIGH:0.50;DESCEND_MID:0.30` |

### Symbolic Motion & Semantics
| Column | Meaning |
|---|---|
| `MotionPhases (Symbolic phase sequence, e.g. STILL_LOW;ASCEND_MID)` | Deduped sequence for readability |
| `SemanticRole (Heuristic high-level role: Player/Enemy/Projectile/etc.)` | Guess from name/category/behavior |

### Interactions & Events
| Column | Meaning |
|---|---|
| `InteractionSummary (Close-approach / touch events with other objects)` | e.g., `moves close to Crate01; touches DoorA.` |
| `MaxSpeedEvent (Notable peak speed moment)` | Peak speed snippet |
| `LargestVerticalChangeEvent (Notable vertical excursion)` | Largest vertical span snippet |

### Narratives
| Column | Meaning |
|---|---|
| `PathSummary (Short, deterministic natural-language description)` | Stats → concise text |
| `NarrativeSummary (Compact narrative description using motion, phases & interactions)` | Story-like, per-object |

---

## Usage

```bash
python trim_scene_v8.py \
  --static "../Data/Slime Rancher/worldGenerated_static.csv" \
  --frames "../Data/Slime Rancher/worldGenerated_frames.csv" \
  --movement-threshold 0.2 \
  --vis-threshold 0.05 \
  --top-k 300 \
  --outdir "./trajectories"
```

--movement-threshold (world units): minimum total path length to keep an object (default 0.2)
--vis-threshold: minimum visibility ratio to keep an object (default 0.05)
--top-k: keep the k most “important” objects by an internal score (0 = keep all; default 300)
--outdir: output directory (default: ./trajectories)

## Result: trajectories/trajectories.csv

For LLM prompting:

Feed the entire CSV for short scenes, or
Select top-N rows by importance / role (e.g., Player, Enemy, Projectile), and
Ask the model to narrate chronologically using PhaseTimeline and PathTrace, and to use InteractionSummary for relationships.

# Sample prompt is available in the repo.

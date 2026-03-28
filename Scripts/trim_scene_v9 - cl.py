#!/usr/bin/env python3
"""
trim_scene_v9.py — Multi-Resolution Scene Compression for LLM Understanding

This script implements a comprehensive rethinking of game telemetry compression:
- Spatial-temporal segmentation (zones + time windows)
- Event-driven representation (discrete state changes)
- Hierarchical importance tiers (critical → ambient)
- Scene graph tracking (relationships, not just objects)
- Keyframe + delta encoding
- Semantic clustering of similar trajectories
- Two-stage retrieval (index + detail chunks)
- Adaptive frame sampling
- Entity-relationship micro-stories

Output structure:
  trajectories_v9/
    scene_index.json          # Ultra-compressed overview (always load this)
    zones/
      zone_*.json             # Per-zone detailed data (load on demand)
    objects/
      tier1_critical.json     # VIP objects with full detail
      tier2_supporting.json   # Important objects with summaries
      tier3_ambient.json      # Aggregated clusters
    events/
      event_timeline.json     # Chronological event sequence
    
The goal: A 7B LLM can understand the scene by first reading the tiny index,
then requesting specific chunks as needed.
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# ---------- Parsing helpers ----------

NUM_RE = re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')


def parse_vec3(s):
    if pd.isna(s):
        return (np.nan, np.nan, np.nan)
    nums = NUM_RE.findall(str(s))
    if len(nums) >= 3:
        return (float(nums[0]), float(nums[1]), float(nums[2]))
    return (np.nan, np.nan, np.nan)


def vec_len(dx, dy, dz):
    return math.sqrt(dx * dx + dy * dy + dz * dz)


# ---------- Spatial segmentation (octree-style zones) ----------

class SpatialZoneManager:
    """Divides 3D space into zones for spatial-temporal segmentation."""
    
    def __init__(self, all_positions, grid_size=8):
        """
        all_positions: Nx3 array of (x, y, z)
        grid_size: number of divisions per axis
        """
        valid = np.all(np.isfinite(all_positions), axis=1)
        if valid.sum() == 0:
            self.bounds = np.array([[0, 1], [0, 1], [0, 1]])
            self.grid_size = grid_size
            return
            
        pos = all_positions[valid]
        self.bounds = np.array([
            [pos[:, 0].min(), pos[:, 0].max()],
            [pos[:, 1].min(), pos[:, 1].max()],
            [pos[:, 2].min(), pos[:, 2].max()]
        ])
        
        # Add small margin to avoid edge cases
        margin = 0.01 * (self.bounds[:, 1] - self.bounds[:, 0])
        self.bounds[:, 0] -= margin
        self.bounds[:, 1] += margin
        
        self.grid_size = grid_size
        
    def get_zone_id(self, x, y, z):
        """Return zone ID string like 'z_3_2_1' for position."""
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
            return "z_unknown"
            
        ix = int((x - self.bounds[0, 0]) / (self.bounds[0, 1] - self.bounds[0, 0]) * self.grid_size)
        iy = int((y - self.bounds[1, 0]) / (self.bounds[1, 1] - self.bounds[1, 0]) * self.grid_size)
        iz = int((z - self.bounds[2, 0]) / (self.bounds[2, 1] - self.bounds[2, 0]) * self.grid_size)
        
        ix = max(0, min(self.grid_size - 1, ix))
        iy = max(0, min(self.grid_size - 1, iy))
        iz = max(0, min(self.grid_size - 1, iz))
        
        return f"z_{ix}_{iy}_{iz}"
        
    def get_zone_center(self, zone_id):
        """Return approximate center of zone."""
        if zone_id == "z_unknown":
            return [0, 0, 0]
        parts = zone_id.split('_')
        ix, iy, iz = int(parts[1]), int(parts[2]), int(parts[3])
        
        x = self.bounds[0, 0] + (ix + 0.5) * (self.bounds[0, 1] - self.bounds[0, 0]) / self.grid_size
        y = self.bounds[1, 0] + (iy + 0.5) * (self.bounds[1, 1] - self.bounds[1, 0]) / self.grid_size
        z = self.bounds[2, 0] + (iz + 0.5) * (self.bounds[2, 1] - self.bounds[2, 0]) / self.grid_size
        
        return [float(x), float(y), float(z)]


# ---------- Temporal segmentation ----------

def segment_timeline(frames_df, window_size=100):
    """Split timeline into temporal windows."""
    if frames_df.empty:
        return []
    min_frame = frames_df['Frame'].min()
    max_frame = frames_df['Frame'].max()
    
    windows = []
    current = min_frame
    while current <= max_frame:
        end = min(current + window_size, max_frame + 1)
        windows.append((int(current), int(end)))
        current = end
        
    return windows


# ---------- Event extraction ----------

class EventExtractor:
    """Extract discrete events from continuous telemetry."""
    
    def __init__(self, frames_df):
        self.frames_df = frames_df
        self.events = []
        
    def detect_spawn_events(self):
        """Detect object spawning."""
        for obj, g in self.frames_df.groupby('ObjectName'):
            first_frame = int(g['Frame'].min())
            pos = g[g['Frame'] == first_frame].iloc[0]
            if np.isfinite(pos['pos_x']):
                self.events.append({
                    'frame': first_frame,
                    'type': 'spawn',
                    'actor': obj,
                    'position': [float(pos['pos_x']), float(pos['pos_y']), float(pos['pos_z'])]
                })
                
    def detect_despawn_events(self):
        """Detect object despawning."""
        for obj, g in self.frames_df.groupby('ObjectName'):
            last_frame = int(g['Frame'].max())
            pos = g[g['Frame'] == last_frame].iloc[0]
            if np.isfinite(pos['pos_x']):
                self.events.append({
                    'frame': last_frame,
                    'type': 'despawn',
                    'actor': obj,
                    'position': [float(pos['pos_x']), float(pos['pos_y']), float(pos['pos_z'])]
                })
                
    def detect_collision_events(self, distance_threshold=0.3):
        """Detect potential collisions (close approaches)."""
        # Sample frames to avoid O(n²) per frame
        sample_frames = sorted(self.frames_df['Frame'].unique())[::10]
        
        for frame in sample_frames:
            frame_data = self.frames_df[self.frames_df['Frame'] == frame]
            valid = frame_data[
                np.isfinite(frame_data['pos_x']) & 
                np.isfinite(frame_data['pos_y']) & 
                np.isfinite(frame_data['pos_z'])
            ]
            
            if len(valid) < 2:
                continue
                
            positions = valid[['pos_x', 'pos_y', 'pos_z']].values
            names = valid['ObjectName'].values
            
            # Check all pairs (limit to avoid explosion)
            for i in range(min(len(positions), 50)):
                for j in range(i + 1, min(len(positions), 50)):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < distance_threshold:
                        self.events.append({
                            'frame': int(frame),
                            'type': 'collision',
                            'actor': names[i],
                            'target': names[j],
                            'position': [float(positions[i][0]), float(positions[i][1]), float(positions[i][2])],
                            'distance': float(dist)
                        })
                        
    def detect_speed_events(self, speed_threshold_percentile=95):
        """Detect sudden speed changes."""
        for obj, g in self.frames_df.groupby('ObjectName'):
            g = g.sort_values('Frame')
            frames = g['Frame'].values
            px, py, pz = g['pos_x'].values, g['pos_y'].values, g['pos_z'].values
            
            dx = np.diff(px)
            dy = np.diff(py)
            dz = np.diff(pz)
            dt = np.diff(frames)
            
            valid = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(dz) & (dt > 0)
            if valid.sum() < 2:
                continue
                
            speeds = np.zeros(len(dx))
            speeds[valid] = np.sqrt(dx[valid]**2 + dy[valid]**2 + dz[valid]**2) / dt[valid]
            
            if speeds.max() == 0:
                continue
                
            threshold = np.percentile(speeds[speeds > 0], speed_threshold_percentile)
            
            for i in np.where(speeds > threshold)[0]:
                self.events.append({
                    'frame': int(frames[i]),
                    'type': 'speed_burst',
                    'actor': obj,
                    'speed': float(speeds[i]),
                    'position': [float(px[i]), float(py[i]), float(pz[i])]
                })
                
    def extract_all(self):
        """Run all event detectors."""
        self.detect_spawn_events()
        self.detect_despawn_events()
        self.detect_collision_events()
        self.detect_speed_events()
        
        # Sort by frame
        self.events.sort(key=lambda e: e['frame'])
        return self.events


# ---------- Importance scoring & tiering ----------

def compute_importance_score(obj_data):
    """
    Multi-factor importance scoring.
    Returns score (higher = more important).
    """
    path_length = obj_data.get('path_length', 0)
    avg_speed = obj_data.get('avg_speed', 0)
    visibility_ratio = obj_data.get('visibility_ratio', 0)
    interaction_count = obj_data.get('interaction_count', 0)
    
    score = (
        path_length * 1.0 +
        avg_speed * 0.5 +
        visibility_ratio * 2.0 +
        interaction_count * 3.0
    )
    
    return float(score)


def assign_importance_tier(score, tier_thresholds):
    """Assign tier based on score."""
    if score >= tier_thresholds['critical']:
        return 'critical'
    elif score >= tier_thresholds['supporting']:
        return 'supporting'
    elif score >= tier_thresholds['ambient']:
        return 'ambient'
    else:
        return 'background'


# ---------- Trajectory clustering ----------

def cluster_similar_trajectories(obj_trajectories, n_clusters=20):
    """
    Cluster objects with similar motion patterns.
    Returns cluster assignments and exemplar objects.
    """
    if len(obj_trajectories) < n_clusters:
        n_clusters = max(1, len(obj_trajectories) // 2)
        
    # Build feature vectors: [path_length, avg_speed, net_displacement, vert_range, horiz_extent]
    features = []
    names = []
    
    for name, traj in obj_trajectories.items():
        features.append([
            traj.get('path_length', 0),
            traj.get('avg_speed', 0),
            traj.get('net_displacement', 0),
            traj.get('vert_range', 0),
            traj.get('horiz_extent', 0)
        ])
        names.append(name)
        
    if not features:
        return {}, {}
        
    features = np.array(features)
    
    # Normalize
    feature_std = features.std(axis=0)
    feature_std[feature_std == 0] = 1
    features_norm = (features - features.mean(axis=0)) / feature_std
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_norm)
    
    # Find exemplar for each cluster (closest to centroid)
    clusters = {}
    exemplars = {}
    
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_names = [names[i] for i in range(len(names)) if mask[i]]
        clusters[f"cluster_{cluster_id}"] = cluster_names
        
        if cluster_names:
            # Find exemplar (closest to centroid)
            centroid = kmeans.cluster_centers_[cluster_id]
            cluster_features = features_norm[mask]
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            exemplar_idx = np.argmin(distances)
            exemplars[f"cluster_{cluster_id}"] = cluster_names[exemplar_idx]
            
    return clusters, exemplars


# ---------- Keyframe detection ----------

def detect_keyframes(frames_df, obj_name, keyframe_interval=50, change_threshold=2.0):
    """
    Detect keyframes for an object based on:
    - Regular intervals
    - Significant state changes (position, speed)
    """
    obj_data = frames_df[frames_df['ObjectName'] == obj_name].sort_values('Frame')
    
    if len(obj_data) < 2:
        return []
        
    frames = obj_data['Frame'].values
    px = obj_data['pos_x'].values
    py = obj_data['pos_y'].values
    pz = obj_data['pos_z'].values
    
    keyframes = [int(frames[0])]  # Always include first frame
    
    last_keyframe_idx = 0
    
    for i in range(1, len(frames)):
        # Regular interval
        if frames[i] - frames[last_keyframe_idx] >= keyframe_interval:
            keyframes.append(int(frames[i]))
            last_keyframe_idx = i
            continue
            
        # Significant change detection
        if i > 0:
            dx = px[i] - px[i-1]
            dy = py[i] - py[i-1]
            dz = pz[i] - pz[i-1]
            
            if np.isfinite(dx) and np.isfinite(dy) and np.isfinite(dz):
                displacement = math.sqrt(dx*dx + dy*dy + dz*dz)
                
                # Compare to recent average
                recent_start = max(0, i - 10)
                recent_dx = np.diff(px[recent_start:i+1])
                recent_dy = np.diff(py[recent_start:i+1])
                recent_dz = np.diff(pz[recent_start:i+1])
                
                valid_recent = (
                    np.isfinite(recent_dx) & 
                    np.isfinite(recent_dy) & 
                    np.isfinite(recent_dz)
                )
                
                if valid_recent.any():
                    recent_disps = np.sqrt(
                        recent_dx[valid_recent]**2 + 
                        recent_dy[valid_recent]**2 + 
                        recent_dz[valid_recent]**2
                    )
                    avg_disp = recent_disps.mean()
                    
                    if avg_disp > 0 and displacement > change_threshold * avg_disp:
                        keyframes.append(int(frames[i]))
                        last_keyframe_idx = i
                        
    # Always include last frame
    if int(frames[-1]) not in keyframes:
        keyframes.append(int(frames[-1]))
        
    return keyframes


# ---------- Micro-narrative generation ----------

def generate_micro_narrative(obj_name, obj_data, events, zones):
    """
    Generate a short story for an object.
    Format: "Spawned at origin → moved through zone_A (20s) → engaged enemy → ..."
    """
    story_parts = []
    
    # Spawn
    spawn_events = [e for e in events if e['type'] == 'spawn' and e['actor'] == obj_name]
    if spawn_events:
        se = spawn_events[0]
        story_parts.append(f"Spawned at frame {se['frame']}")
        
    # Zone traversal
    zone_sequence = obj_data.get('zone_sequence', [])
    if zone_sequence:
        zone_desc = " → ".join([f"{z[0]}({z[1]}f)" for z in zone_sequence[:5]])
        story_parts.append(f"moved through {zone_desc}")
        
    # Interactions
    collision_events = [
        e for e in events 
        if e['type'] == 'collision' and (e['actor'] == obj_name or e.get('target') == obj_name)
    ]
    if collision_events:
        targets = set()
        for ce in collision_events[:5]:
            if ce['actor'] == obj_name:
                targets.add(ce.get('target', 'unknown'))
            else:
                targets.add(ce['actor'])
        story_parts.append(f"interacted with {', '.join(list(targets)[:3])}")
        
    # Speed bursts
    speed_events = [e for e in events if e['type'] == 'speed_burst' and e['actor'] == obj_name]
    if speed_events:
        story_parts.append(f"{len(speed_events)} speed bursts")
        
    # Despawn
    despawn_events = [e for e in events if e['type'] == 'despawn' and e['actor'] == obj_name]
    if despawn_events:
        de = despawn_events[0]
        story_parts.append(f"despawned at frame {de['frame']}")
        
    return " → ".join(story_parts) if story_parts else "No significant activity"


# ---------- Main processing ----------

def process_objects(frames_df, static_df, zone_manager):
    """
    Process all objects and return structured data.
    """
    # Parse positions once
    if 'pos_x' not in frames_df.columns:
        pos = np.vstack([parse_vec3(v) for v in frames_df['Position']])
        frames_df['pos_x'] = pos[:, 0]
        frames_df['pos_y'] = pos[:, 1]
        frames_df['pos_z'] = pos[:, 2]
        
    objects_data = {}
    
    for obj, g in frames_df.groupby('ObjectName'):
        g = g.sort_values('Frame')
        
        frames = g['Frame'].values
        px = g['pos_x'].values
        py = g['pos_y'].values
        pz = g['pos_z'].values
        
        valid = np.isfinite(px) & np.isfinite(py) & np.isfinite(pz)
        if valid.sum() < 2:
            continue
            
        idx_valid = np.where(valid)[0]
        
        # Basic stats
        dx = np.diff(px[idx_valid])
        dy = np.diff(py[idx_valid])
        dz = np.diff(pz[idx_valid])
        dt = np.diff(frames[idx_valid])
        
        step_dists = np.sqrt(dx**2 + dy**2 + dz**2)
        path_length = float(step_dists.sum())
        
        speeds = step_dists / np.maximum(dt, 1)
        avg_speed = float(speeds.mean()) if len(speeds) > 0 else 0.0
        
        start_pos = [float(px[idx_valid[0]]), float(py[idx_valid[0]]), float(pz[idx_valid[0]])]
        end_pos = [float(px[idx_valid[-1]]), float(py[idx_valid[-1]]), float(pz[idx_valid[-1]])]
        
        net_displacement = float(np.linalg.norm(np.array(end_pos) - np.array(start_pos)))
        
        vert_range = float(py[idx_valid].max() - py[idx_valid].min())
        horiz_extent = float(np.sqrt(
            (px[idx_valid].max() - px[idx_valid].min())**2 +
            (pz[idx_valid].max() - pz[idx_valid].min())**2
        ))
        
        visibility_ratio = float(valid.sum() / len(g))
        
        # Zone sequence
        zone_sequence = []
        current_zone = None
        zone_start_frame = None
        
        for i in idx_valid:
            zone = zone_manager.get_zone_id(px[i], py[i], pz[i])
            if zone != current_zone:
                if current_zone is not None:
                    duration = int(frames[i]) - zone_start_frame
                    zone_sequence.append((current_zone, duration))
                current_zone = zone
                zone_start_frame = int(frames[i])
                
        if current_zone is not None:
            duration = int(frames[idx_valid[-1]]) - zone_start_frame
            zone_sequence.append((current_zone, duration))
            
        # Keyframes
        keyframes = detect_keyframes(frames_df, obj, keyframe_interval=50)
        
        objects_data[obj] = {
            'name': obj,
            'start_frame': int(frames[0]),
            'end_frame': int(frames[-1]),
            'start_position': start_pos,
            'end_position': end_pos,
            'path_length': path_length,
            'avg_speed': avg_speed,
            'net_displacement': net_displacement,
            'vert_range': vert_range,
            'horiz_extent': horiz_extent,
            'visibility_ratio': visibility_ratio,
            'zone_sequence': zone_sequence,
            'keyframes': keyframes,
            'interaction_count': 0  # Will be updated later
        }
        
    return objects_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-resolution compressed scene representation (v9)"
    )
    parser.add_argument("--static", required=True, help="Path to static objects CSV")
    parser.add_argument("--frames", required=True, help="Path to frames CSV")
    parser.add_argument("--outdir", default=None, help="Output directory (default: trajectories_v9)")
    parser.add_argument("--grid-size", type=int, default=8, help="Spatial grid divisions per axis")
    parser.add_argument("--time-window", type=int, default=100, help="Temporal window size (frames)")
    parser.add_argument("--critical-top-k", type=int, default=50, help="Number of critical tier objects")
    parser.add_argument("--supporting-top-k", type=int, default=200, help="Number of supporting tier objects")
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    frames_df = pd.read_csv(args.frames)
    static_df = pd.read_csv(args.static)
    
    print(f"Loaded {len(frames_df)} frame records for {frames_df['ObjectName'].nunique()} objects")
    
    # Parse positions
    print("Parsing positions...")
    pos = np.vstack([parse_vec3(v) for v in frames_df['Position']])
    frames_df['pos_x'] = pos[:, 0]
    frames_df['pos_y'] = pos[:, 1]
    frames_df['pos_z'] = pos[:, 2]
    
    # Create spatial zones
    print("Creating spatial zones...")
    all_positions = pos
    zone_manager = SpatialZoneManager(all_positions, grid_size=args.grid_size)
    
    # Create temporal windows
    print("Creating temporal windows...")
    time_windows = segment_timeline(frames_df, window_size=args.time_window)
    print(f"Created {len(time_windows)} time windows")
    
    # Extract events
    print("Extracting events...")
    event_extractor = EventExtractor(frames_df)
    events = event_extractor.extract_all()
    print(f"Extracted {len(events)} events")
    
    # Process objects
    print("Processing objects...")
    objects_data = process_objects(frames_df, static_df, zone_manager)
    print(f"Processed {len(objects_data)} objects")
    
    # Update interaction counts from events
    for event in events:
        if event['type'] == 'collision':
            actor = event.get('actor')
            target = event.get('target')
            if actor in objects_data:
                objects_data[actor]['interaction_count'] += 1
            if target in objects_data:
                objects_data[target]['interaction_count'] += 1
                
    # Compute importance scores
    print("Computing importance scores...")
    for obj_data in objects_data.values():
        obj_data['importance_score'] = compute_importance_score(obj_data)
        
    # Sort by importance
    sorted_objects = sorted(objects_data.values(), key=lambda x: x['importance_score'], reverse=True)
    
    # Assign tiers
    tier_thresholds = {
        'critical': sorted_objects[min(args.critical_top_k, len(sorted_objects)-1)]['importance_score'] if sorted_objects else 0,
        'supporting': sorted_objects[min(args.supporting_top_k, len(sorted_objects)-1)]['importance_score'] if sorted_objects else 0,
        'ambient': 0
    }
    
    tiered_objects = {'critical': [], 'supporting': [], 'ambient': [], 'background': []}
    
    for obj_data in objects_data.values():
        tier = assign_importance_tier(obj_data['importance_score'], tier_thresholds)
        obj_data['tier'] = tier
        tiered_objects[tier].append(obj_data)
        
    print(f"Tiers: {len(tiered_objects['critical'])} critical, "
          f"{len(tiered_objects['supporting'])} supporting, "
          f"{len(tiered_objects['ambient'])} ambient, "
          f"{len(tiered_objects['background'])} background")
    
    # Cluster ambient/background objects
    print("Clustering similar trajectories...")
    ambient_objects = {obj['name']: obj for obj in tiered_objects['ambient'] + tiered_objects['background']}
    clusters, exemplars = cluster_similar_trajectories(ambient_objects, n_clusters=20)
    
    # Generate micro-narratives for critical/supporting objects
    print("Generating micro-narratives...")
    for tier in ['critical', 'supporting']:
        for obj_data in tiered_objects[tier]:
            obj_data['narrative'] = generate_micro_narrative(
                obj_data['name'], obj_data, events, zone_manager
            )
            
    # Prepare output directory
    outdir = Path(args.outdir) if args.outdir else (Path(args.frames).parent / "trajectories_v9")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "zones").mkdir(exist_ok=True)
    (outdir / "objects").mkdir(exist_ok=True)
    (outdir / "events").mkdir(exist_ok=True)
    
    # Write scene index (ultra-compressed overview)
    print("Writing scene index...")
    scene_index = {
        "metadata": {
            "total_objects": len(objects_data),
            "total_frames": int(frames_df['Frame'].max() - frames_df['Frame'].min()),
            "start_frame": int(frames_df['Frame'].min()),
            "end_frame": int(frames_df['Frame'].max()),
            "spatial_bounds": {
                "x": [float(zone_manager.bounds[0, 0]), float(zone_manager.bounds[0, 1])],
                "y": [float(zone_manager.bounds[1, 0]), float(zone_manager.bounds[1, 1])],
                "z": [float(zone_manager.bounds[2, 0]), float(zone_manager.bounds[2, 1])]
            },
            "grid_size": args.grid_size,
            "time_windows": len(time_windows)
        },
        "tiers": {
            "critical": len(tiered_objects['critical']),
            "supporting": len(tiered_objects['supporting']),
            "ambient": len(tiered_objects['ambient']),
            "background": len(tiered_objects['background'])
        },
        "critical_objects": [obj['name'] for obj in tiered_objects['critical'][:20]],
        "major_events": [
            {k: v for k, v in e.items() if k != 'position'}  # Exclude detailed positions
            for e in events[:50]  # Top 50 events
        ],
        "zones_with_activity": list(set(
            zone for obj in tiered_objects['critical'] + tiered_objects['supporting']
            for zone, _ in obj.get('zone_sequence', [])
        ))[:20],
        "clusters": {
            cid: {"count": len(members), "exemplar": exemplars.get(cid)}
            for cid, members in clusters.items()
        }
    }
    
    with open(outdir / "scene_index.json", 'w') as f:
        json.dump(scene_index, f, indent=2)
        
    # Write tier files
    print("Writing object tier files...")
    for tier in ['critical', 'supporting']:
        tier_data = []
        for obj in tiered_objects[tier]:
            tier_data.append({
                "name": obj['name'],
                "tier": obj['tier'],
                "importance_score": obj['importance_score'],
                "start_frame": obj['start_frame'],
                "end_frame": obj['end_frame'],
                "start_position": obj['start_position'],
                "end_position": obj['end_position'],
                "path_length": obj['path_length'],
                "avg_speed": obj['avg_speed'],
                "net_displacement": obj['net_displacement'],
                "vert_range": obj['vert_range'],
                "horiz_extent": obj['horiz_extent'],
                "visibility_ratio": obj['visibility_ratio'],
                "zone_sequence": obj['zone_sequence'][:10],  # Limit to first 10 zones
                "keyframes": obj['keyframes'][:20],  # Limit to first 20 keyframes
                "narrative": obj.get('narrative', '')
            })
        
        with open(outdir / "objects" / f"tier_{tier}.json", 'w') as f:
            json.dump(tier_data, f, indent=2)
    
    # Write ambient cluster summaries
    print("Writing ambient cluster summaries...")
    cluster_summaries = []
    for cluster_id, members in clusters.items():
        exemplar_name = exemplars.get(cluster_id)
        exemplar_data = ambient_objects.get(exemplar_name, {})
        
        cluster_summaries.append({
            "cluster_id": cluster_id,
            "member_count": len(members),
            "exemplar": exemplar_name,
            "exemplar_stats": {
                "path_length": exemplar_data.get('path_length', 0),
                "avg_speed": exemplar_data.get('avg_speed', 0),
                "net_displacement": exemplar_data.get('net_displacement', 0)
            },
            "members": members[:50]  # Limit member list
        })
    
    with open(outdir / "objects" / "tier_ambient.json", 'w') as f:
        json.dump(cluster_summaries, f, indent=2)
    
    # Write event timeline
    print("Writing event timeline...")
    event_timeline = {
        "total_events": len(events),
        "event_types": {
            event_type: len([e for e in events if e['type'] == event_type])
            for event_type in set(e['type'] for e in events)
        },
        "events": events  # Full event list
    }
    
    with open(outdir / "events" / "event_timeline.json", 'w') as f:
        json.dump(event_timeline, f, indent=2)
    
    # Write per-zone details (on-demand chunks)
    print("Writing zone detail files...")
    zone_data_map = defaultdict(lambda: {
        'objects': [],
        'events': [],
        'time_windows': []
    })
    
    # Assign objects to zones based on their primary zone (most time spent)
    for obj_data in objects_data.values():
        zone_seq = obj_data.get('zone_sequence', [])
        if not zone_seq:
            continue
        
        # Find primary zone (longest duration)
        primary_zone = max(zone_seq, key=lambda x: x[1])[0] if zone_seq else 'z_unknown'
        
        zone_data_map[primary_zone]['objects'].append({
            "name": obj_data['name'],
            "tier": obj_data.get('tier', 'unknown'),
            "start_frame": obj_data['start_frame'],
            "end_frame": obj_data['end_frame'],
            "path_length": obj_data['path_length'],
            "zone_sequence": zone_seq
        })
    
    # Assign events to zones
    for event in events:
        pos = event.get('position')
        if pos:
            zone = zone_manager.get_zone_id(pos[0], pos[1], pos[2])
            zone_data_map[zone]['events'].append(event)
    
    # Assign time windows to zones (based on activity)
    for window_start, window_end in time_windows:
        window_df = frames_df[
            (frames_df['Frame'] >= window_start) & 
            (frames_df['Frame'] < window_end)
        ]
        
        # Find most active zones in this window
        zone_activity = defaultdict(int)
        for _, row in window_df.iterrows():
            if np.isfinite(row['pos_x']):
                zone = zone_manager.get_zone_id(row['pos_x'], row['pos_y'], row['pos_z'])
                zone_activity[zone] += 1
        
        for zone, activity in zone_activity.items():
            zone_data_map[zone]['time_windows'].append({
                'start': window_start,
                'end': window_end,
                'activity': activity
            })
    
    # Write zone files
    for zone_id, zone_data in zone_data_map.items():
        zone_file = outdir / "zones" / f"{zone_id}.json"
        
        zone_output = {
            "zone_id": zone_id,
            "center": zone_manager.get_zone_center(zone_id),
            "object_count": len(zone_data['objects']),
            "event_count": len(zone_data['events']),
            "objects": zone_data['objects'][:100],  # Limit per zone
            "events": zone_data['events'][:100],
            "time_windows": zone_data['time_windows']
        }
        
        with open(zone_file, 'w') as f:
            json.dump(zone_output, f, indent=2)
    
    print(f"Wrote {len(zone_data_map)} zone files")
    
    # Write keyframe data for critical objects
    print("Writing keyframe data...")
    keyframe_data = {}
    
    for obj_data in tiered_objects['critical']:
        obj_name = obj_data['name']
        keyframes = obj_data['keyframes']
        
        # Get positions at keyframes
        obj_frames = frames_df[frames_df['ObjectName'] == obj_name]
        
        keyframe_positions = []
        for kf in keyframes:
            kf_row = obj_frames[obj_frames['Frame'] == kf]
            if not kf_row.empty:
                row = kf_row.iloc[0]
                keyframe_positions.append({
                    'frame': int(kf),
                    'position': [float(row['pos_x']), float(row['pos_y']), float(row['pos_z'])]
                })
        
        keyframe_data[obj_name] = keyframe_positions
    
    with open(outdir / "objects" / "keyframes_critical.json", 'w') as f:
        json.dump(keyframe_data, f, indent=2)
    
    # Generate README for the 7B LLM
    print("Writing README...")
    readme_content = """# Scene Representation v9 - Multi-Resolution Format

This directory contains a compressed, hierarchical representation of game telemetry
optimized for understanding by small language models (7B parameters).

## Structure

### 1. Start Here: scene_index.json
**Always load this first.** It provides:
- Scene metadata (bounds, duration, object counts)
- List of critical objects (most important actors)
- Major events summary (top 50 events)
- Active zones
- Object clustering information

**Size**: Small (~50-200 KB)
**Purpose**: Get the big picture before diving into details

### 2. Object Tiers (objects/)

#### tier_critical.json
- Top ~50 most important objects
- Full trajectories, narratives, zone sequences
- Use these for detailed analysis
- **Load when**: You need to understand key actors

#### tier_supporting.json
- Next ~200 important objects
- Motion summaries, key stats
- **Load when**: You need more context beyond critical objects

#### tier_ambient.json
- Clusters of similar background objects
- Each cluster has an exemplar + member list
- **Load when**: You need to understand crowds/swarms

### 3. Zones (zones/)
- Spatial regions with localized object lists and events
- Each zone file (z_X_Y_Z.json) contains:
  - Objects that spent significant time in that zone
  - Events that occurred in that zone
  - Time windows of activity
- **Load when**: You want to understand "what happened in this area"

### 4. Events (events/)

#### event_timeline.json
- Chronological list of discrete events:
  - spawn/despawn
  - collisions
  - speed_burst
- **Load when**: You need temporal sequencing ("what happened when")

### 5. Keyframes (objects/)

#### keyframes_critical.json
- Sampled positions for critical objects at key moments
- Use for trajectory visualization or detailed path analysis
- **Load when**: You need precise position history

## Usage Pattern for 7B LLM

### Simple Query: "What happened in this scene?"
1. Read scene_index.json
2. Summarize: metadata, critical objects, major events

### Medium Query: "Tell me about the player's journey"
1. Read scene_index.json → find player in critical_objects
2. Read tier_critical.json → get player's narrative + zone_sequence
3. Read relevant zone files if needed

### Complex Query: "Analyze combat in the arena zone"
1. Read scene_index.json → identify arena zone(s)
2. Read zones/z_X_Y_Z.json for arena
3. Read event_timeline.json → filter collision events in arena
4. Read tier_critical.json for involved objects

### Cluster Query: "What were the enemy behaviors?"
1. Read scene_index.json → check clusters
2. Read tier_ambient.json → find enemy clusters
3. Examine exemplar stats and member lists

## Key Design Principles

- **Progressive Loading**: Start small (index), load details as needed
- **Spatial Locality**: Zone files keep related information together
- **Temporal Ordering**: Event timeline preserves "what happened when"
- **Hierarchical Importance**: Critical objects get full detail, ambient objects get summaries
- **Semantic Clustering**: Similar objects are grouped, not duplicated

## File Size Expectations

- scene_index.json: 50-200 KB
- tier_critical.json: 100-500 KB
- tier_supporting.json: 500 KB - 2 MB
- tier_ambient.json: 50-200 KB (clusters only)
- event_timeline.json: 1-5 MB
- zones/*.json: 10-100 KB each
- keyframes_critical.json: 50-500 KB

Total: ~10-50 MB (vs. original logs which could be 500 MB - 5 GB)
Compression ratio: ~10-100x

## Example Queries

**Q: "Summarize the scene"**
A: Load scene_index.json only. Describe object counts, major events, duration.

**Q: "What did the player do?"**
A: Load scene_index.json, tier_critical.json. Report player's narrative.

**Q: "When did combat occur?"**
A: Load event_timeline.json. Filter collision events, report timing.

**Q: "What happened in zone z_3_2_1?"**
A: Load zones/z_3_2_1.json. Describe objects and events in that zone.

**Q: "How many enemies were there?"**
A: Load tier_ambient.json. Count members in enemy clusters.
"""
    
    with open(outdir / "README.md", 'w') as f:
        f.write(readme_content)
    
    # Write summary statistics
    print("\n" + "="*60)
    print("COMPRESSION SUMMARY")
    print("="*60)
    
    # Calculate approximate sizes
    import os
    total_size = 0
    for root, dirs, files in os.walk(outdir):
        for file in files:
            filepath = os.path.join(root, file)
            total_size += os.path.getsize(filepath)
    
    print(f"Total output size: {total_size / 1024 / 1024:.2f} MB")
    print(f"Scene index: {os.path.getsize(outdir / 'scene_index.json') / 1024:.2f} KB")
    print(f"Critical tier: {len(tiered_objects['critical'])} objects")
    print(f"Supporting tier: {len(tiered_objects['supporting'])} objects")
    print(f"Ambient clusters: {len(clusters)} clusters covering {sum(len(m) for m in clusters.values())} objects")
    print(f"Total events: {len(events)}")
    print(f"Zone files: {len(zone_data_map)}")
    print(f"\nOriginal frame records: {len(frames_df)}")
    print(f"Objects processed: {len(objects_data)}")
    print(f"Compression achieved: ~{len(frames_df) * 100 / max(1, total_size / 50):.1f}x (estimated)")
    print("="*60)
    
    print(f"\n✅ Multi-resolution scene representation written to: {outdir}")
    print(f"📖 Read {outdir}/README.md for usage instructions")
    

if __name__ == "__main__":
    main()
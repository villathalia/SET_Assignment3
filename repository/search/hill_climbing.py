"""
Assignment 3 — Scenario-Based Testing of an RL Agent (Hill Climbing)

You MUST implement:
    - compute_objectives_from_time_series
    - compute_fitness
    - mutate_config
    - hill_climb

DO NOT change function signatures.
You MAY add helper functions.

Goal
----
Find a scenario (environment configuration) that triggers a collision.
If you cannot trigger a collision, minimize the minimum distance between the ego
vehicle and any other vehicle across the episode.

Black-box requirement
---------------------
Your evaluation must rely only on observable behavior during execution:
- crashed flag from the environment
- time-series data returned by run_episode (positions, lane_id, etc.)
No internal policy/model details beyond calling policy(obs, info).
"""

import copy
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

from envs.highway_env_utils import run_episode


# ============================================================
# 1) OBJECTIVES FROM TIME SERIES
# ============================================================

def compute_objectives_from_time_series(time_series: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute your objective values from the recorded time-series.

    The time_series is a list of frames. Each frame typically contains:
      - frame["crashed"]: bool
      - frame["ego"]: dict or None, e.g. {"pos":[x,y], "lane_id":..., "length":..., "width":...}
      - frame["others"]: list of dicts with positions, lane_id, etc.

    Minimum requirements (suggested):
      - crash_count: 1 if any collision happened, else 0
      - min_distance: minimum distance between ego and any other vehicle over time (float)

    Return a dictionary, e.g.:
        {
          "crash_count": 0 or 1,
          "min_distance": float
        }

    NOTE: If you want, you can add more objectives (lane-specific distances, time-to-crash, etc.)
    but keep the keys above at least.
    """
    # TODO (students)
    NEAR_MISS_THRESHOLD = 2.0  # Threshold distance for near-miss counting (assumed pragmatically)

    # Initialize default values
    crash_count = 0
    min_distance = float("inf")
    min_same_lane_distance = float("inf")
    min_adjacent_lane_distance = float("inf")

    near_miss_count = 0

    lane_changes = 0
    prev_lane_id = None
    lane_id_missing = 0

    max_deceleration = 0.0
    max_acceleration = 0.0
    prev_speed = None

    for frame in time_series:
        # Check for crash
        if frame.get("crashed", False):
            crash_count = 1
        
        # Compute Distance
        ego_data = frame.get("ego")
        if not ego_data:
            continue

        ego_pos = np.array(ego_data["pos"]) # Use numpy array for easy math
        others = frame.get("others", [])

        # Find closest car in this specific frame
        frame_min_dist = float("inf")
        
        for other in others:
            if not other: continue
            other_pos = np.array(other["pos"])
            
            # Calculate Distance (Replaces helper._euclidean)
            dist = np.linalg.norm(ego_pos - other_pos)

            # Update Global Minimum
            if dist < min_distance:
                min_distance = dist
            
            # Update Frame Minimum
            if dist < frame_min_dist:
                frame_min_dist = dist

        # Count Near Misses (Helps guide the search before a crash happens)
        if frame_min_dist < NEAR_MISS_THRESHOLD:
            near_miss_count += 1

    # Safety fallback if no cars were seen
    if min_distance == float("inf"):
        min_distance = 100.0

    return {
        "crash_count": int(crash_count),
        "min_distance": float(min_distance),
        "near_miss_count": int(near_miss_count)
    }

def compute_fitness(objectives: Dict[str, Any]) -> float:
    """
    Convert objectives into ONE scalar fitness value to MINIMIZE.

    Requirement:
    - Any crashing scenario must be strictly better than any non-crashing scenario.

    Examples:
    - If crash_count==1: fitness = -1 (best)
    - Else: fitness = min_distance (smaller is better)

    You can design a more refined scalarization if desired.
    """
    # TODO (students)
    raise NotImplementedError


# ============================================================
# 2) MUTATION / NEIGHBOR GENERATION
# ============================================================

def mutate_config(
    cfg: Dict[str, Any],
    param_spec: Dict[str, Any],
    rng: np.random.Generator
) -> Dict[str, Any]:
    """
    Generate ONE neighbor configuration by mutating the current scenario.

    Inputs:
      - cfg: current scenario dict (e.g., vehicles_count, initial_spacing, ego_spacing, initial_lane_id)
      - param_spec: search space bounds, types (int/float), min/max
      - rng: random generator

    Requirements:
      - Do NOT modify cfg in-place (return a copy).
      - Keep mutated values within [min, max] from param_spec.
      - If you mutate lanes_count, keep initial_lane_id valid (0..lanes_count-1).

    Students can implement:
      - single-parameter mutation (recommended baseline)
      - multiple-parameter mutation
      - adaptive step sizes, etc.
    """
    new_cfg = copy.deepcopy(cfg)
    
    # 15% chance to mutate EVERYTHING (Big jump), 85% chance to mutate ONE thing (Small step)
    if rng.random() < 0.15:
        params_to_mutate = list(param_spec.keys())
    else:
        params_to_mutate = [rng.choice(list(param_spec.keys()))]
    
    for param in params_to_mutate:
        spec = param_spec[param]
        current_val = new_cfg[param]
    
        # MUTATE INTEGER (e.g., vehicles_count)
        if spec["type"] == "int":
            # Dynamic step size logic
            range_val = max(1, spec["max"] - spec["min"])
            sigma = range_val * 0.1
            delta = rng.normal(0, sigma)
            
            # Ensure at least 1 unit move if delta is non-zero
            if abs(delta) > 0.1:
                mutation = int(np.sign(delta) * np.ceil(abs(delta))) 
            else:
                mutation = 0
            
            # Special case: Always try to move at least 1 if it's the only param we picked
            if mutation == 0 and len(params_to_mutate) == 1:
                mutation = int(rng.choice([-1, 1]))
                
            new_val = int(current_val + mutation)
            new_cfg[param] = int(np.clip(new_val, spec["min"], spec["max"]))

        # MUTATE FLOAT (e.g., spacing)
        elif spec["type"] == "float":
            range_width = spec["max"] - spec["min"]
            sigma = range_width * 0.05 # 5% perturbation
            new_val = float(rng.normal(current_val, sigma))
            new_cfg[param] = float(np.clip(new_val, spec["min"], spec["max"]))

    # CONSTRAINT: Keep initial_lane_id valid if lanes_count changed [cite: 57]
    if "lanes_count" in new_cfg and "initial_lane_id" in new_cfg:
        allowed_max = new_cfg["lanes_count"] - 1
        new_cfg["initial_lane_id"] = int(np.clip(new_cfg["initial_lane_id"], 0, allowed_max))
    
    return new_cfg


# ============================================================
# 3) HILL CLIMBING SEARCH
# ============================================================

def hill_climb(
    env_id: str,
    base_cfg: Dict[str, Any],
    param_spec: Dict[str, Any],
    policy,
    defaults: Dict[str, Any],
    seed: int = 0,
    iterations: int = 100,
    neighbors_per_iter: int = 10,
) -> Dict[str, Any]:
    """
    Hill climbing loop.

    You should:
      1) Start from an initial scenario (base_cfg or random sample).
      2) Evaluate it by running:
            crashed, ts = run_episode(env_id, cfg, policy, defaults, seed_base)
         Then compute objectives + fitness.
      3) For each iteration:
            - Generate neighbors_per_iter neighbors using mutate_config
            - Evaluate each neighbor
            - Select the best neighbor
            - Accept it if it improves fitness (or implement another acceptance rule)
            - Optionally stop early if a crash is found
      4) Return the best scenario found and enough info to reproduce.

    Return dict MUST contain at least:
        {
          "best_cfg": Dict[str, Any],
          "best_objectives": Dict[str, Any],
          "best_fitness": float,
          "best_seed_base": int,
          "history": List[float]
        }

    Optional but useful:
        - "best_time_series": ts
        - "evaluations": int
    """
    rng = np.random.default_rng(seed)
    
    # 1. Initialization: Sample random start or use base_cfg
    # (Using random start helps find crashes faster usually)
    current_cfg = copy.deepcopy(base_cfg) 
    
    # Initial Eval
    seed_base = int(rng.integers(1e9))
    crashed, ts = run_episode(env_id, current_cfg, policy, defaults, seed_base)
    obj = compute_objectives_from_time_series(ts)
    cur_fit = compute_fitness(obj)

    # Tracking Best
    best_cfg = copy.deepcopy(current_cfg)
    best_obj = dict(obj)
    best_fit = float(cur_fit)
    best_seed_base = seed_base
    history = [best_fit]

    print(f"Start Fitness: {best_fit:.4f} | Crash: {obj['crash_count']}")

    # Early exit if we started with a crash
    if crashed or obj['crash_count'] == 1:
        print("!!! INITIAL SCENARIO CRASHED !!!")
        return {
            "best_cfg": best_cfg, "best_objectives": best_obj, 
            "best_fitness": best_fit, "best_seed_base": best_seed_base, 
            "history": history, "evaluations": 1
        }

    evals = 1

    # 2. Main Loop
    for it in range(iterations):
        
        # Track best neighbor in this batch
        best_neighbor_cfg = None
        best_neighbor_fit = float('inf')
        best_neighbor_obj = None
        best_neighbor_seed = None

        for n in range(neighbors_per_iter):
            # Generate Neighbor
            cand_cfg = mutate_config(current_cfg, param_spec, rng)
            cand_seed = int(rng.integers(1e9))
            
            # Evaluate
            c_crashed, c_ts = run_episode(env_id, cand_cfg, policy, defaults, cand_seed)
            c_obj = compute_objectives_from_time_series(c_ts)
            c_fit = compute_fitness(c_obj)
            evals += 1

            # Check for immediate crash discovery
            if c_crashed or c_obj['crash_count'] == 1:
                print(f"!!! CRASH FOUND at Iteration {it}, Neighbor {n} !!!")
                return {
                    "best_cfg": cand_cfg,
                    "best_objectives": c_obj,
                    "best_fitness": c_fit,
                    "best_seed_base": cand_seed,
                    "history": history + [c_fit],
                    "evaluations": evals
                }

            # Update best neighbor
            if c_fit < best_neighbor_fit:
                best_neighbor_fit = c_fit
                best_neighbor_cfg = cand_cfg
                best_neighbor_obj = c_obj
                best_neighbor_seed = cand_seed

        # 3. Selection (Steepest Ascent/Descent)
        # If the best neighbor is better than our CURRENT position, move there
        if best_neighbor_fit < cur_fit:
            print(f"Iter {it}: Improved fitness {cur_fit:.4f} -> {best_neighbor_fit:.4f}")
            current_cfg = best_neighbor_cfg
            cur_fit = best_neighbor_fit
            # Also update global best if strictly better
            if cur_fit < best_fit:
                best_cfg = current_cfg
                best_fit = cur_fit
                best_obj = best_neighbor_obj
                best_seed_base = best_neighbor_seed
        else:
            print(f"Iter {it}: Stuck. Best neighbor ({best_neighbor_fit:.4f}) not better than current ({cur_fit:.4f})")

        history.append(best_fit)

    return {
        "best_cfg": best_cfg,
        "best_objectives": best_obj,
        "best_fitness": best_fit,
        "best_seed_base": best_seed_base,
        "history": history,
        "evaluations": evals
    }
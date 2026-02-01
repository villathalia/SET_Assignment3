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
from search import helper_HC as helper


# ============================================================
# 1) OBJECTIVES FROM TIME SERIES
# ============================================================
def compute_objectives_from_time_series(
    time_series: List[Dict[str, Any]],
) -> Dict[str, Any]:
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

    Extra addded objectives:
      - lane_changes: number of ego lane changes
      - max_deceleration: max speed drop between consecutive frames (per step)
      - max_acceleration: max speed increase between consecutive frames (per step)
      - near_miss_count: #frames with distance < NEAR_MISS_THRESHOLD
      - min_same_lane_distance: min distance to others in same lane (if lane_id available)
      - min_adjacent_lane_distance: min distance to others in adjacent lanes (if lane_id available)
      - lane_id_missing_ratio: fraction of frames where ego lane_id is None
    """
    NEAR_MISS_THRESHOLD = (
        2.0  # Threshold distance for near-miss counting (assumed pragmatically)
    )

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
        if frame.get("crashed", False):
            crash_count = 1

        ego = frame.get("ego", None)
        if not ego:
            continue

        ego_pos = ego.get("pos", None)
        if ego_pos is None or len(ego_pos) < 2:
            continue

        ego_lane = ego.get("lane_id", None)
        if ego_lane is None:
            lane_id_missing += 1

        # lane changes count
        if (
            prev_lane_id is not None
            and ego_lane is not None
            and ego_lane != prev_lane_id
        ):
            lane_changes += 1
        if ego_lane is not None:
            prev_lane_id = ego_lane

        # acceleration/deceleration proxy from speed
        speed = ego.get("speed", None)
        if speed is not None:
            speed = float(speed)
            if prev_speed is not None:
                delta = speed - prev_speed
                if delta > max_acceleration:
                    max_acceleration = delta
                if -delta > max_deceleration:
                    max_deceleration = -delta
            prev_speed = speed

        # distances to others
        others = frame.get("others", []) or []
        frame_min = float("inf")

        for other in others:
            if not other:
                continue
            other_pos = other.get("pos", None)
            if other_pos is None or len(other_pos) < 2:
                continue

            d = helper._euclidean(ego_pos, other_pos)

            # global min
            if d < min_distance:
                min_distance = d
            if d < frame_min:
                frame_min = d

            # lane-specific mins
            other_lane = other.get("lane_id", None)
            if ego_lane is not None and other_lane is not None:
                if other_lane == ego_lane:
                    if d < min_same_lane_distance:
                        min_same_lane_distance = d
                elif abs(int(other_lane) - int(ego_lane)) == 1:
                    if d < min_adjacent_lane_distance:
                        min_adjacent_lane_distance = d

        # near miss count based on per-frame closest approach
        if frame_min < NEAR_MISS_THRESHOLD:
            near_miss_count += 1

    # handle "never computed" cases
    if min_distance == float("inf"):
        min_distance = 1e9
    if min_same_lane_distance == float("inf"):
        min_same_lane_distance = 1e9
    if min_adjacent_lane_distance == float("inf"):
        min_adjacent_lane_distance = 1e9

    lane_id_missing_ratio = lane_id_missing / max(1, len(time_series))

    return {
        # required objectives
        "crash_count": int(crash_count),
        "min_distance": float(min_distance),
        # extra objectives
        "lane_changes": int(lane_changes),
        "max_deceleration": float(max_deceleration),
        "max_acceleration": float(max_acceleration),
        "near_miss_count": int(near_miss_count),
        "min_same_lane_distance": float(min_same_lane_distance),
        "min_adjacent_lane_distance": float(min_adjacent_lane_distance),
        "lane_id_missing_ratio": float(lane_id_missing_ratio),
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
    crash = int(objectives.get("crash_count", 0))
    min_dist = float(objectives.get("min_distance", 1e9))
    near_miss = int(objectives.get("near_miss_count", 0))

    if crash == 1:
        # Penalize higher min_dist even during crash to find "dead center" collisions
        return -1e6

    # Smaller distance and higher near_miss counts result in better (lower) fitness
    return min_dist - (0.1 * near_miss)


# ============================================================
# 2) MUTATION / NEIGHBOR GENERATION
# ============================================================


def mutate_config(
    cfg: Dict[str, Any], param_spec: Dict[str, Any], rng: np.random.Generator
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
    # Requirement: Do NOT modify cfg in-place (return a copy) [cite: 51, 54]
    new_cfg = copy.deepcopy(cfg)

    # Increase exploration: 30% chance to mutate multiple params,
    # otherwise pick just one to maintain direction.
    # Pick a random parameter from the search space to mutate [cite: 22, 53]
    if rng.random() < 0.15:
        params_to_mutate = list(param_spec.keys())
        logger.info(f"Starting multi-parameter mutation on: {params_to_mutate}")
    else:
        params_to_mutate = [rng.choice(list(param_spec.keys()))]
        logger.info(f"Starting single-parameter mutation on: {params_to_mutate[0]}")

    for param in params_to_mutate:
        spec = param_spec[param]
        current_val = new_cfg[param]

        # Apply mutation based on the parameter type [cite: 58, 59]
        if spec["type"] == "int":
            # Dynamic range-based integer shifts
            range_val = max(1, spec["max"] - spec["min"])
            sigma = range_val * 0.1
            delta = rng.normal(0, sigma)
            # Ensure at least 1 unit move if delta is present
            mutation = (
                int(np.sign(delta) * np.ceil(abs(delta))) if abs(delta) > 0.1 else 0
            )

            # Special case for vehicle count to allow small creeps or larger jumps
            if param == "vehicles_count" and mutation == 0:
                mutation = int(rng.choice([-1, 1]))

            raw_new_val = int(current_val + mutation)
            clipped_val = int(np.clip(raw_new_val, spec["min"], spec["max"]))
            new_cfg[param] = clipped_val
            logger.info(
                f"  MUTATE {param}: {current_val} -> {clipped_val} "
                f"(raw: {raw_new_val}, delta: {mutation}, bounds: [{spec['min']},{spec['max']}])"
            )

        elif spec["type"] == "float":
            # Float mutation with 5% sigma for fine-tuning min_distance
            range_width = spec["max"] - spec["min"]
            sigma = range_width * 0.05
            raw_new_val = float(rng.normal(current_val, sigma))
            clipped_val = float(np.clip(raw_new_val, spec["min"], spec["max"]))
            new_cfg[param] = clipped_val

            logger.info(
                f"  MUTATE {param}: {current_val:.4f} -> {clipped_val:.4f} "
                f"(raw: {raw_new_val:.4f}, sigma: {sigma:.4f}, bounds: [{spec['min']},{spec['max']}])"
            )

    # Requirement: If lanes_count is mutated, keep initial_lane_id valid [cite: 57, 59]
    # lane constraint ONLY when relevant
    old_lane = int(cfg.get("initial_lane_id", 0))
    lanes = int(new_cfg.get("lanes_count", cfg.get("lanes_count")))
    lane_id_spec_max = int(param_spec["initial_lane_id"]["max"])  # usually 4
    allowed_max = min(
        lane_id_spec_max, lanes - 1
    )  # cannot be higher than lane count - 1

    new_cfg["initial_lane_id"] = int(
        np.clip(new_cfg["initial_lane_id"], 0, allowed_max)
    )
    if new_cfg["initial_lane_id"] != old_lane:
        logger.info(
            f"  CONSTRAINT: Adjusted initial_lane_id {old_lane} -> {new_cfg['initial_lane_id']} for {lanes} lanes"
        )

    if new_cfg == cfg:
        logger.info(
            "  NO-OP: mutation produced identical config after clipping/constraints"
        )

    return new_cfg


# ============================================================
# 3) HILL CLIMBING SEARCH
# ============================================================
import logging

# Configure logging to save to a file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("hc_search.log"),  # Saves to this file
        logging.StreamHandler(),  # Still prints to console
    ],
)
logger = logging.getLogger(__name__)


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

    # Stopping criteria  for HC
    PATIENCE = 15  # stop if no global-best improvement for this many iterations
    MIN_DELTA = 1e-6  # improvement threshold to count as "real" improvement
    fitness_per_iteration = []
    rng = np.random.default_rng(seed)

    # Initialize from RANDOM start scenario
    current_cfg = helper.sample_random_cfg(base_cfg, param_spec, rng)

    # Evaluate initial solution (seed_base used for reproducibility)
    seed_base = int(rng.integers(1e9))
    crashed, ts = run_episode(env_id, current_cfg, policy, defaults, seed_base)
    obj = compute_objectives_from_time_series(ts)
    cur_fit = compute_fitness(obj)

    evaluations = 1  # initial evaluation

    # Best-so-far tracking
    best_cfg = copy.deepcopy(current_cfg)
    best_obj = dict(obj)
    best_fit = float(cur_fit)
    best_seed_base = seed_base
    best_time_series = ts

    # History: best-so-far fitness
    history = [best_fit]
    fitness_per_iteration.append(best_fit)

    # Log the start of the search
    logger.info(f"Starting Hill Climbing search. Initial fitness: {best_fit}")

    # Early stop if initial is already a crash
    if crashed or best_obj.get("crash_count", 0) == 1:
        logger.info(f"Collision found at evaluation {evaluations} (initial scenario).")
        return {
            "best_cfg": best_cfg,
            "best_objectives": best_obj,
            "best_fitness": float(best_fit),
            "best_seed_base": int(best_seed_base),
            "history": history,
            "best_time_series": best_time_series,
            "evaluations": int(evaluations),
            "iterations_executed": 0,
            "fitness_per_iteration": fitness_per_iteration,
        }

    # Plateau tracking
    no_improve_iters = 0

    # Hill Climbing loop
    for n in range(iterations):
        # Best neighbor in this iteration
        logger.info(f"--- HC Iteration {n+1}/{iterations} ---")
        logger.info(
            f"Parent cfg: {current_cfg} | cur_fit={cur_fit:.6f} | best_fit={best_fit:.6f}"
        )

        best_n_cfg = None
        best_n_obj = None
        best_n_fit = None
        best_n_seed = None
        best_n_ts = None

        # Generate and evaluate neighbors
        for i in range(neighbors_per_iter):
            logger.info(f"Neighbor Iteration {i+1} ")

            # Generate neighbours from the random starting scenario
            cand_cfg = mutate_config(current_cfg, param_spec, rng)

            # Evaluate neighbor, using a fixed seed
            cand_seed = int(rng.integers(1e9))
            crashed, ts = run_episode(env_id, cand_cfg, policy, defaults, cand_seed)

            cand_obj = compute_objectives_from_time_series(ts)

            # Compute fitness
            cand_fit = float(compute_fitness(cand_obj))
            evaluations += 1

            logger.info(f"Eval {evaluations}: Candidate Fitness={cand_fit:.4f}")

            # Keep best neighbor (minimize fitness)
            logger.info(
                f"Candidate fitness: {cand_fit}, best neighbor fitness: {best_n_fit}"
            )
            if (best_n_fit is None) or (cand_fit < best_n_fit):
                best_n_cfg = copy.deepcopy(cand_cfg)
                best_n_obj = dict(cand_obj)
                best_n_fit = float(cand_fit)
                best_n_seed = int(cand_seed)
                best_n_ts = ts

            # Early stop as soon as crash is found
            if crashed or cand_obj.get("crash_count", 0) == 1:
                logger.info(f"Collision found at evaluation {evaluations}!")
                best_cfg = copy.deepcopy(cand_cfg)
                best_obj = dict(cand_obj)
                best_fit = float(cand_fit)
                best_seed_base = int(cand_seed)
                best_time_series = ts
                fitness_per_iteration.append(best_fit)

                return {
                    "best_cfg": best_cfg,
                    "best_objectives": best_obj,
                    "best_fitness": float(best_fit),
                    "best_seed_base": int(best_seed_base),
                    "history": history,
                    "best_time_series": best_time_series,
                    "evaluations": int(evaluations),
                    "iterations_executed": n + 1,
                    "fitness_per_iteration": fitness_per_iteration,
                }

        # if no neighbors were evaluated, stop
        logger.info(
            f"Best neighbor fitness after all neighbour iteration {i+1} after for loop: {best_n_fit}"
        )
        if best_n_fit is None:
            break

        # Accept the best neighbor if it improves CURRENT fitness
        logger.info(f"cur_fit after for loop: {cur_fit}")
        if best_n_fit < cur_fit:
            current_cfg = copy.deepcopy(best_n_cfg)
            obj = dict(best_n_obj)
            cur_fit = float(best_n_fit)

        # Update GLOBAL best if improved enough. Threshold to avoid tiny/noisy improvements
        improved_global = False
        logger.info(f"best_fit & curr_fit before global update: {best_fit}, {cur_fit}")
        if cur_fit < best_fit - MIN_DELTA:
            best_cfg = copy.deepcopy(current_cfg)
            best_obj = dict(obj)
            best_fit = float(cur_fit)
            best_seed_base = int(best_n_seed)
            best_time_series = best_n_ts
            improved_global = True
            logger.info(f"New global best fitness: {best_fit}")

        fitness_per_iteration.append(best_fit)
        # Stagnation or plateau is reached. Need the count for stopping
        if improved_global:
            no_improve_iters = 0
        else:
            no_improve_iters += 1

        logger.info(f"No improvement iterations: {no_improve_iters}")
        # Track best-so-far fitness after each iteration
        if len(history) == 0 or history[-1] != best_fit:
            # logger.info(f"Appending to history: {history[-1]}, {best_fit}")
            history.append(best_fit)

        # Stop if we're stuck on a plateau
        if no_improve_iters >= PATIENCE:
            logger.info("Early stopping due to plateau.")
            break

    # Return best solution
    return {
        "best_cfg": best_cfg,
        "best_objectives": best_obj,
        "best_fitness": float(best_fit),
        "best_seed_base": int(best_seed_base),
        "history": history,
        "best_time_series": best_time_series,
        "evaluations": int(evaluations),
        "iterations_executed": n + 1,
        "fitness_per_iteration": fitness_per_iteration,
    }

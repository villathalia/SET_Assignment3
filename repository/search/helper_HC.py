import math
import json
import csv
from typing import Dict, Any, Optional, List
from envs.highway_env_utils import record_video_episode
from types import SimpleNamespace

# ============================================================
# 1) Calculate distance between two points
# ============================================================
def _euclidean(p1, p2) -> float:
    dx = float(p1[0]) - float(p2[0])
    dy = float(p1[1]) - float(p2[1])
    return math.sqrt(dx * dx + dy * dy)

# ============================================================
# 2) Random Scenario from Search Space for HC SEARCH
# ============================================================

def sample_random_cfg(base_cfg, param_spec, rng):
    """
    Reuse Random Search's sampling logic (ScenarioSearch.sample_random_config)
    so Hill Climbing starts from the exact same "random start" distribution.
    """
    from search.base_search import ScenarioSearch

    tmp = SimpleNamespace(base_cfg=base_cfg, param_spec=param_spec)
    return ScenarioSearch.sample_random_config(tmp, rng)

# ============================================================
# 3) Save Recording 
# ============================================================
def record_best_if_crash(result: Dict[str, Any], env_id: str, policy, defaults, out_dir: str = "videos"):
    """
    Record a video for the best scenario only if it crashed.
    """
    if result["best_objectives"].get("crash_count", 0) != 1:
        return False

    cfg = result["best_cfg"]
    seed_base = int(result["best_seed_base"])
    record_video_episode(env_id, cfg, policy, defaults, seed_base, out_dir=out_dir)
    return True

# ============================================================
# 4) Save Objectives Interactions and Results in CSV
# ============================================================
def save_objectives_csv(result: Dict[str, Any], filepath: str = "hc_best_objectives.csv"):
    """
    Save best objectives as a 2-column CSV: metric,value
    """
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in result["best_objectives"].items():
            w.writerow([k, v])

# ============================================================
# 5) Save Objectives Interactions and Results in JSON
# ============================================================
def save_result_json(result: Dict[str, Any], filepath: str = "hc_result.json"):
    """
    Save the full result (including cfg, history, objectives) to JSON for grading/debugging.
    """
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)

# ============================================================
# 6) Plot Fitness History
# ==========================================================
def plot_history(result: Dict[str, Any], filepath: str = "hc_fitness_history.png"):
    """
    Plot the best-so-far fitness history.
    """
    import matplotlib.pyplot as plt

    fitness = result["fitness_per_iteration"]
    plt.figure()
    x = range(1, len(fitness) + 1)   # Iterations start at 1
    y = fitness     # Fitness values

    plt.plot(x, y,marker='o')

    plt.xlabel("Iteration")
    plt.ylabel("Best-so-far fitness (lower is better)")
    plt.title("Hill Climbing Fitness Improvement")

    plt.xticks(x)          
    plt.ticklabel_format(style="plain", axis="y")
    plt.grid(True)
    plt.savefig(filepath, dpi=200)
    plt.close()
    

 
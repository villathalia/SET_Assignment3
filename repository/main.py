from config.search_space import param_spec, base_cfg
from policies.pretrained_policy import load_pretrained_policy
from envs.highway_env_utils import make_env
from search.hill_climbing import hill_climb
from search.random_search import RandomSearch
import search.helper_HC as helper


def main():
    env_id = "highway-fast-v0"
    policy = load_pretrained_policy("agents/model")
    env, defaults = make_env(env_id)
    print(env, policy, defaults, sep="\n")

    # --- Random Search ---
    """
    search = RandomSearch(env_id, base_cfg, param_spec, policy, defaults)
    crashes = search.run_search(n_scenarios=50, seed=11)

    print(f"Random Search found {len(crashes)} crashes.")
    if crashes:
        print(crashes)

    # Save Random Search results if available
    if hasattr(search, "best_cfg") and search.best_cfg is not None:
        rs_result = {
            "best_cfg": search.best_cfg,
            "best_objectives": search.best_objectives,
            "best_fitness": search.best_fitness,
            "history": search.history,
            "evaluations": 50,
            "best_seed_base": getattr(search, "best_seed_base", 0),
        }
        helper.save_result_json(rs_result, "rs_result.json")
        helper.save_objectives_csv(rs_result, "rs_best_objectives.csv")
        helper.plot_history(rs_result, "rs_fitness_history.png")

        if helper.record_best_if_crash(
            rs_result, env_id, policy, defaults, out_dir="videos_rs"
        ):
            print("Recorded best RS crash video.")

    # If RandomSearch.py is not updated, but found crashes
    elif crashes:
        print("RandomSearch class not updated, but crashes were found.")
        first_crash = crashes[0]

        rs_minimal_result = {
            "best_cfg": first_crash["cfg"],
            "best_seed_base": first_crash["seed"],
            "best_objectives": {"crash_count": 1},
        }

        if helper.record_best_if_crash(
            rs_minimal_result, env_id, policy, defaults, out_dir="videos_rs"
        ):
            print("Recorded RS crash video.")
    else:
        print("RS results not tracked. Skipping save.")

    """
    # --- Hill Climbing ---
    hc_result = hill_climb(
        env_id=env_id,
        base_cfg=base_cfg,
        param_spec=param_spec,
        policy=policy,
        defaults=defaults,
        seed=11,
        iterations=100,
        neighbors_per_iter=10,
    )
    print("Hill Climbing finished.")
    hc_result["fitness_per_iteration"] = hc_result["history"]
    print("Best fitness:", hc_result["best_fitness"])
    print("Evaluations:", hc_result["evaluations"])
    print("Iterations:", hc_result["iterations_executed"])

    hc_crash_count = hc_result["best_objectives"]["crash_count"]
    if hc_crash_count:
        print(f"Hill Climbing found a crashes.")

    # save results and record video if crash
    helper.save_result_json(hc_result, "hc_result.json")
    helper.save_objectives_csv(hc_result, "hc_best_objectives.csv")
    # helper.plot_history(hc_result, "hc_fitness_history.png")
    recorded_hc = helper.record_best_if_crash(
        hc_result, env_id, policy, defaults, out_dir="videos"
    )
    if recorded_hc:
        print("Recorded best crash video.")
    else:
        print("Best scenario did not crash; no video recorded.")


if __name__ == "__main__":
    main()

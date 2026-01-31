from config.search_space import param_spec, base_cfg
from policies.pretrained_policy import load_pretrained_policy
from envs.highway_env_utils import make_env
from search.hill_climbing import hill_climb
from search.random_search import RandomSearch
import search.helper_HC as helper

def main():
    env_id = "highway-fast-v0"
    policy = load_pretrained_policy("agents/model.zip")
    env, defaults = make_env(env_id)

    # --- Random Search ---
   # search = RandomSearch(env_id, base_cfg, param_spec, policy, defaults)
    #crashes = search.run_search(n_scenarios=50, seed=11)

    #print(f"✅ Found {len(crashes)} crashes.")
    #if crashes:
    #print(crashes)

    # --- Hill Climbing ---
    result = hill_climb(
        env_id=env_id,
        base_cfg=base_cfg,
        param_spec=param_spec,
        policy=policy,
        defaults=defaults,
        seed=11,
        iterations=100,
        neighbors_per_iter=10,
    )
    print("✅ Hill Climbing finished.")
    print("Best fitness:", result["best_fitness"])
    print("Evaluations:", result["evaluations"])
    print("Iterations:", result["iterations_executed"])

    # save results and record video if crash
    helper.save_result_json(result, "hc_result.json")
    helper.save_objectives_csv(result, "hc_best_objectives.csv")
    helper.plot_history(result, "hc_fitness_history.png")
    recorded = helper.record_best_if_crash(result, env_id, policy, defaults, out_dir="videos")
    if recorded:
        print("Recorded best crash video.")
    else:
        print("Best scenario did not crash; no video recorded.")

if __name__ == "__main__":
    main()
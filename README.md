# SET_Assignment3

**Testing Self Driving car Systems**

This project demonstrates how we can search collision sceraio in a systematic way in the search space using HillClimbing technique and find out possible collision scenerios when we change single/multiple environment configurations and they interact together in a hypothetical road simulation.

The goal is to verify:

1) Automatically identify environment configurations that trigger collisions in the RL-controlled vehicle.

2) Expose near-collision and risky interaction patterns that may not surface through random testing.

3) Assess agent robustness using only observable execution data, without access to internal model details.

4) Verify Test suite effectiveness via mutation testingEnsure consistent and repeatable evaluation of scenarios through controlled randomness and deterministic fitness computation.

## Repository Structure

We have put our file for Hill Climbing under:
```
repository/
└──search/
    └── hill_climbing.py
    └── helper_HC.py
└──main.py
└──hc_result.json
└──hc_fitness_history.png

```
## File Descriptions

1) hill_climbing.py

Core module implementing the hill-climbing search algorithm contains:

- compute_objectives_from_time_series() – extracts safety-related objectives from episode traces

- compute_fitness() – converts objectives into a single scalar fitness value

- mutate_config() – generates neighbor scenarios via parameter mutation

- hill_climb() – main optimization loop that searches for collision-inducing scenarios

This file contains the full search logic.

2) helper_HC.py

Utility functions used by hill_climbing.py contains:

- random scenario sampling
- distance computation helpers
- save results of HC & plot fitness history
- video recording helpers

This file supports the search implementation.

3) main.py

Entry point for running scenario-based testing. It Handles:

- loading the environment and pre-trained PPO policy
- run the random search and record its crashes & parameters.
- calling hill_climb() with user-specified settings and save the results & recording.

This file orchestrates the experiment.

## Running the Project

Run the HillLCimbing search from **main.py**:

```bash
python main.py
```

This will:
- Load the pre-trained PPO policy.
- Initialize a random highway scenario for HC search.
- First search using random search and produce collisions.
- Next, Iteratively mutate the scenario using hill climbing to search for failure-inducing configurations.
- Evaluate each candidate by running the environment and computing a fitness score.
- Stop early if a collision is found or when progress plateau.
- Record the best-found collision (if any) to `hc_videos/` as an MP4 file.
- Using hc_helper.py, save the results in a json file 'hc_result.json' and created a fitness vs number of iterations graph 'hc_fitness_history.png' to show the trend of the HC run.

import gymnasium as gym
import numpy as np


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk

env_id = "Walker2d-v5"  # Example environment ID
seed = 42
capture_video = False
run_name = "crossq_example_run"

envs = gym.vector.SyncVectorEnv(
    [make_env(env_id, seed, 0, capture_video, run_name) for i in range(3)]
)

state, _ = envs.reset(seed=seed)

while True:
    #actions = envs.action_space.sample()  # Sample a random action
    actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
    next_obs, rewards, terminations, truncations, infos = envs.step(actions)
    


    # if "episode" in infos:
    #     print(infos)
    #print(np.logical_or(terminations, truncations))
    for idx, trunc in enumerate(truncations):
        if trunc:
           print(truncations)
           print(infos)
           #real_next_obs[idx] = infos["final_observation"][idx]
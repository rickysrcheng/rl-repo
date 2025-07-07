import gymnasium as gym
import numpy as np
import pprint


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
    [make_env(env_id, seed, 0, capture_video, run_name) for i in range(3)],
    autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
)

state, _ = envs.reset(seed=seed)

while True:
    #actions = envs.action_space.sample()  # Sample a random action
    actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
    next_obs, rewards, terminations, truncations, infos = envs.step(actions)
    

    real_next_state = np.zeros_like(next_obs)
    if "_final_info" in infos:  
        #pprint.pprint(infos)  # Print the infos dictionary for debugging
        for i in range(envs.num_envs):
            if infos['_final_info'][i]:
                print(f"Env {i} finished with reward: {infos['final_info']['episode']['r'][i]}")
                print(infos['final_obs'][i])
                real_next_state[i] = infos['final_obs'][i]
                print(real_next_state)
            # Optionally reset the environment if needed
            # envs.reset_at(i)
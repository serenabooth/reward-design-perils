import gym
from gym_hungry_thirsty.envs import hungry_thirsty_reward_fns
from gym_hungry_thirsty.envs.hungry_thirsty_env import Available_Actions
from copy import deepcopy

env = gym.make('gym_hungry_thirsty:hungry-thirsty-v0', size=(4,4))
reward = None

NUM_EPISODES = 2

for _ in range(NUM_EPISODES):
    episode_trajectory = []
    obs = env.reset(new_water_food_loc=True)
    episode_metadata = {
                        "food_loc": env.food_loc,
                        "water_loc": env.water_loc
                       }
    while True:
        action = Available_Actions.random()
        # print (action)

        # construct the trajectory for a return fn score
        episode_trajectory.append((deepcopy(obs), action))
        new_obs, _, done, info = env.step(action)

        reward = hungry_thirsty_reward_fns.sparse_reward_fn(state=obs,
                                                            action=action,
                                                            new_state=new_obs)
        obs = new_obs
        # print("reward", reward)

        if done:
            env.playback(trajectory=episode_trajectory, episode_metadata=episode_metadata)
            episode_trajectory = []
            break

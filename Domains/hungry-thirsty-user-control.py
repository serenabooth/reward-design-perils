import gym
import time
import sys
import pickle

from gym_hungry_thirsty.envs.hungry_thirsty_env import Available_Actions
from copy import deepcopy
size = (4,4)
env = gym.make('gym_hungry_thirsty:hungry-thirsty-v0', size=size)
reward = None
return_val = None

NUM_EPISODES = 1
score = 0

for _ in range(NUM_EPISODES):
    obs = env.reset(food_loc=(0,0), water_loc=(size[0]-1,0))
    trajectory = []

    while True:
        if not obs["hungry"]:
            score += 1
        env.render(score=score)
        time.sleep(0.2)

        print("What action should I take?\n"
              "w,a,s,d: up, left, right, down\n"
              "e: eat\n"
              "r: drink")
        try:
            action = input()
        except KeyboardInterrupt:
            sys.exit(0)

        if action == "w":
            action = Available_Actions.UP
        if action == "a":
            action = Available_Actions.LEFT
        if action == "s":
            action = Available_Actions.DOWN
        if action == "d":
            action = Available_Actions.RIGHT
        if action == "e":
            action = Available_Actions.EAT
        if action == "r":
            action = Available_Actions.DRINK
        # action = Available_Actions.random()

        trajectory.append((deepcopy(obs), action))

        # construct the trajectory for a return fn score
        new_obs, _, done, info = env.step(action)

        obs = new_obs

        if done:
            with open("Assets/user-control.txt", 'wb') as file:
                pickle.dump(trajectory, file)
            print("Resetting the environment")
            obs = env.reset()

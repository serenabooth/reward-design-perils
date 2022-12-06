import time
import numpy as np
import gym
from gym_hungry_thirsty.envs.hungry_thirsty_reward_fns import *
import random


class Value_Iteration:
    """
    Value Iteration solver
    """
    def __init__(self, env, reward_fn=None):
        """
        Initalize the Value Iteration class

        :param env: openai gym env
        :param reward_fn: fn, f(s, a, s') -> float
        """
        self.env = env
        self.policy = {}

        if type(self.env.observation_space) is gym.spaces.discrete.Discrete:
            self.v_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        else:
            self.v_table = self.env.construct_value_table()

        if "hungry-thirsty" in env.spec.id:
            self.env.reset()
            if reward_fn != None:
                self.reward_fn = reward_fn
            else:
                self.reward_fn = sparse_reward_fn
        else:
            self.environment_metadata = {}
            self.env.reset()

    def get_action(self, state):
        """
        Get the action to select
        :param state:
        :return:
        """
        state_hash = self.env.hash_lookup(state)
        return self.policy[state_hash][0]

    def compute_state_value(self, state, gamma):
        """
        Perform the value iteration update for a given state.

        :param state: a string corresponding to a state
        :param gamma: float, the discount factor
        :return: None
        """
        actions = self.env.get_available_actions(state)
        v_s = None
        for action in actions:
            v_s_temp = 0
            available_transitions_and_probs = self.env.get_available_transitions_and_probs(state,
                                                                                           action,
                                                                                           reward_fn=self.reward_fn)

            for prob, reward, new_state in available_transitions_and_probs:
                new_state_hash = self.env.hash_lookup(new_state)
                v_s_temp += prob * (reward + (gamma * self.v_table[new_state_hash]))

            if v_s is None or v_s < v_s_temp:
                v_s = v_s_temp
            else:
                pass #v_s is unchanged

        return v_s

    def value_iteration_training(self, theta, gamma):
        """
        update self.v_table to consist of a sufficiently accurate value table

        :param theta: float, the VI end criterion
        :param gamma: float, the discount factor
        :return: None
        """
        while True:
            delta = 0
            states = list(self.v_table.keys())

            for state in states:
                v = self.v_table[state]
                self.v_table[state] = self.compute_state_value(state=state, gamma=gamma)
                delta = max(delta, abs(v - self.v_table[state]))

            if delta < theta:
                break

    def policy_training(self, gamma):
        """
        train the policy

        :param gamma: float, the discount factor
        :return: None
        """
        states = list(self.v_table.keys())
        for state in states:
            # get all available actions
            available_actions = self.env.get_available_actions(state)
            # track which action to take
            current_action_choice = None

            for action in available_actions:
                # what's the value of taking this action?
                action_val = 0
                available_transitions_and_probs = self.env.get_available_transitions_and_probs(state,
                                                                                               action,
                                                                                               reward_fn=self.reward_fn)
                for prob, reward, new_state in available_transitions_and_probs:
                    new_state_hash = self.env.hash_lookup(new_state)
                    action_val += prob * (reward + (gamma * self.v_table[new_state_hash]))

                if current_action_choice is None or action_val > current_action_choice[1]:
                    current_action_choice = (action, action_val)

            self.policy[state] = current_action_choice

    def compute_avg_fitness(self, num_tests, random_seed=None):
        """

        :param self: a VI object
        :param num_tests: int, the number of tests
        :param random_seed: int, optional. Set to make experiments repeatable.
        :return: float, the mean cumulative fitness
        """
        if random_seed is not None:
            random.seed(random_seed)
        policy_avg_fitness = []
        for idx in range(num_tests):
            done = False
            state = self.env.reset()
            not_hungry_ct = 0
            while not done:
                if not state["hungry"]:
                    not_hungry_ct += 1
                state_hash = self.env.hash_lookup(state)
                action = self.policy[state_hash][0]
                state, _, done, _ = self.env.step(action)

                if done:
                    policy_avg_fitness.append(not_hungry_ct)
        return np.mean(policy_avg_fitness)


def main():
    """ VALUE ITERATION HYPERPARAMETERS """
    gamma = 0.99
    theta = 0.0001

    reward_fn_dict = {'hungry and thirsty': -1, 'hungry and not thirsty': 0.4, 'not hungry and thirsty': -0.15, 'not hungry and not thirsty': 1.0}

    def compute_reward(reward_fn_dict, state):
        """
        Given a state and reward function parameters, return the reward

        :param reward_fn_params: a dict
        :param state: a dictionary state of the form:
            {"position": (x,y),
             "hungry": bool,
             "thirsty": bool}
        :return: float - corresponding to the reward
        """
        if state["hungry"] and state["thirsty"]:
            return reward_fn_dict['hungry and thirsty']
        if state["hungry"] and not state["thirsty"]:
            return reward_fn_dict['hungry and not thirsty']
        if not state["hungry"] and state["thirsty"]:
            return reward_fn_dict['not hungry and thirsty']
        if not state["hungry"] and not state["thirsty"]:
            return reward_fn_dict['not hungry and not thirsty']

        raise Exception("Reward computation failed")

    def reward_fn(state, action, new_state):
        """
        wrapper for compute_reward function, which instantiates reward params

        :param state:
        :param action:
        :param new_state:
        :return:
        """
        return compute_reward(reward_fn_dict=reward_fn_dict, state=state)



    environments = {0: 'gym_hungry_thirsty:hungry-thirsty-v0'}
    print("Choose an environment (0, 1, or 2): " + str(environments))
    env_choice = int(input())
    env = gym.make(environments[env_choice], size=(6,6))
    env.reset(food_loc=(0,0), water_loc=(5,0))
    print ("Create VI obj")
    alg = Value_Iteration(env, reward_fn=reward_fn)
    print ("VI training")
    alg.value_iteration_training(theta=theta, gamma=gamma)
    print ("Policy training")
    alg.policy_training(gamma=gamma)

    while True:
        print("Do you want to see the agent? Y - yes; N - no")
        see_agent = input()
        if not (see_agent == "y" or see_agent == "Y"):
            exit()

        done = False
        state = alg.env.reset()
        # state = alg.env.reset(food_loc=alg.environment_metadata["food_loc"],
        #                       water_loc=alg.environment_metadata["water_loc"])
        fitness = 0
        while not done:
            # show the policy
            if not state['hungry']:
                fitness += 1
            if 'hungry-thirsty' in env.spec.id:
                alg.env.render(score=fitness)
            else:
                alg.env.render()
            time.sleep(0.5)

            state_hash = alg.env.hash_lookup(state)
            action = alg.policy[state_hash][0]
            state, _, done, _ = alg.env.step(action)


if __name__ == "__main__":
    main()


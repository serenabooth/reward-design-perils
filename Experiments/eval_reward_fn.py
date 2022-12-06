import gym
import sys
from gym_hungry_thirsty.envs.hungry_thirsty_env import compute_reward
sys.path.insert(0, 'RL_algorithms')

import A2C
import PPO
import DDQN
import Q_learning


"""
Given an env, hyperparameters, and reward_fn, train an agent and return the cumulative fitness over training
"""


def a2c(env, hyper_params, reward_fn):
    """
    Construct an A2C agent.

    :param env: the OpenAI gym environment
    :param hyper_params: a dict of hyperparameters for the learning alg
    :param reward_fn: a function of (s, a, s')
    :return:
    """
    return A2C.create_a2c_agent(env=env, hyper_params=hyper_params, user_reward_fn=reward_fn, plot_results=False)


def ppo(env, hyper_params, reward_fn):
    """
    Construct a PPO agent.

    :param env: the OpenAI gym environment
    :param hyper_params: a dict of hyperparameters for the learning alg
    :param reward_fn: a function of (s, a, s')
    :return:
    """
    return PPO.create_ppo_agent(env=env, hyper_params=hyper_params, user_reward_fn=reward_fn, plot_results=False)


def ddqn(env, hyper_params, reward_fn):
    """
    Construct a DQN agent.

    :param env: the OpenAI gym environment
    :param hyper_params: a dict of hyperparameters for the learning alg
    :param reward_fn: a function of (s, a, s')
    :return:
    """
    return DDQN.create_ddqn_agent(env=env, hyper_params=hyper_params, user_reward_fn=reward_fn, plot_results=False)


def q_learn(env, hyper_params, reward_fn):
    """
    Construct a Q-learning agent.


    :param env: the OpenAI gym environment
    :param hyper_params: a dict of hyperparameters for the learning alg
    :param reward_fn: a function of (s, a, s')
    :return:
    """
    return Q_learning.create_q_learning_agent(env=env, hyper_params=hyper_params, user_reward_fn=reward_fn)


def eval_reward_fn(alg, hyper_params, reward_fn_params, env_name='hungry-thirsty-v0', num_trials=10, env_size=(4,4)):
    """
    Evaluate a reward function

    :param alg: string, the choice of algorithm (e.g., "A2C", "Q_learn", "DDQN", or "PPO")
    :param hyper_params: a dict of hyperparameters for the learning alg
    :param reward_fn_params: either a dict or list of reward fn parameters
    :param env_name: string, the name of the openai gym environment
    :param num_trials: int, the number of trials to run
    :param env_size: tuple (int, int), the size of the environment
    :return:
    """

    def user_reward_fn(state, action, new_state):
        """
        wrapper for compute_reward function in r(s, a, s') form,
        which instantiates reward params

        :param state: dict, the state
        :param action: int, the action
        :param new_state: dict, the new state
        :return: float, the reward
        """
        return compute_reward(reward_fn_params=reward_fn_params, state=state)

    env = gym.make(env_name, size=env_size)
    if "env_timesteps" in hyper_params.keys():
        env.update_step_limit(hyper_params["env_timesteps"])

    fitness_all_episodes = []

    for _ in range(num_trials):
        env.reset(new_water_food_loc=True)
        if "env_timesteps" in hyper_params.keys():
            env.update_step_limit(hyper_params["env_timesteps"])


        fitness_over_time = None
        if alg == "Q_learn":
            a, fitness_over_time = q_learn(env=env,
                                           hyper_params=hyper_params,
                                           reward_fn=user_reward_fn)
        elif alg == "A2C":
            _, _, fitness_over_time = a2c(env=env,
                                          hyper_params=hyper_params,
                                          reward_fn=user_reward_fn)
        elif alg == "DDQN":
            _, _, _, fitness_over_time = ddqn(env=env,
                                              hyper_params=hyper_params,
                                              reward_fn=user_reward_fn)
        elif alg == "PPO":
            _, _, fitness_over_time = ppo(env=env,
                                          hyper_params=hyper_params,
                                          reward_fn=user_reward_fn)
        fitness_over_time = [x[1] for x in fitness_over_time]
        fitness_all_episodes.append(fitness_over_time)
    return fitness_all_episodes

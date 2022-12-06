import time
import numpy as np
import gym
import random
import pickle
from tqdm import tqdm
from Utils import plot_learning_curve
import gym_hungry_thirsty
from gym_hungry_thirsty.envs.hungry_thirsty_reward_fns import *


class QLearning:
    """
    Q-learning class
    """

    def __init__(self, env, num_episodes, reward_fn=None, epsilon=0.15, record_type="return"):
        """
        Initialize the environment & algorithm

        :param env_name: a string corresponding to the gym environment
        :param num_episodes: an int corresponding to the number of training episodes
        :param reward_fn: the parameters of a reward function, if applicable
        :param record_type: string, indicating what to record (i.e., fitness for hungry thirsty)
        """
        self.env = env
        self.env.reset()
        self.record_type = record_type
        if "hungry-thirsty" in self.env.spec.id:
            self.Q_table = self.env.construct_q_table()
            self.record_type = "fitness"
        else:
            if type(self.env.observation_space) is gym.spaces.discrete.Discrete:
                self.Q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])
            else:
                raise Exception("No target for constructing a Q table")

        self.num_episodes = num_episodes
        # an array for tracking performance over time
        self.learning_performance = []

        self.environment_metadata = {}
        self.env.reset()

        self.epsilon = epsilon

        self.reward_fn = reward_fn

    def q_table_argmax(self, state):
        """
        Find the action which maximizes the q-value for the given state

        :param state: representation varies, but probably a dict or string
        :return: int, the action
        """
        if type(self.Q_table) is np.ndarray:
            return np.argmax(self.Q_table[state, :])
        else:
            action_vals_dict = self.Q_table[self.env.hash_lookup(state)]
            action = max(action_vals_dict, key=action_vals_dict.get)
            return action

    def q_lookup(self, state, action):
        """
        Lookup the q value for a state, action pair

        :param state: representation varies, but probably a dict or string
        :param action: int, the action
        :return: a number, corresponding to the q value
        """
        if type(self.Q_table) is np.ndarray:
            return self.Q_table[state, action]
        else:
            return self.Q_table[self.env.hash_lookup(state)][action]

    def q_update(self, state, action, update_value, alpha_lr):
        """
        Update the Q_table's value for a given state, action pair

        :param state: representation varies, but probably a dict or string
        :param action: int, the action
        :param update_value: the amount to change the value by
        :param alpha_lr: float, the learning rate
        :return:
        """
        if type(self.Q_table) is np.ndarray:
            self.Q_table[state, action] = (1 - alpha_lr) * self.Q_table[state, action] + update_value
        else:
            self.Q_table[self.env.hash_lookup(state)][action] = (1 - alpha_lr) * \
                                                                self.Q_table[self.env.hash_lookup(state)][action] \
                                                                + update_value

    def e_greedy_action_selection(self, state, epsilon):
        """
        Apply epsilon-greedy action selection, and take the step in the environment

        :param state:
        :param epsilon: float, the epsilon for greedy action selection
        :return: (action, new_state, reward, done, info)
            action: int
            new_state: dict
            reward: float
            done: bool
            info: dict
        """
        # use epsilon-greedy action selection
        if random.random() < epsilon:
            action = self.env.get_random_action()
        else:
            action = self.q_table_argmax(state)
        new_state, reward, done, info = self.env.step(action)

        if "hungry-thirsty" in self.env.spec.id and self.reward_fn is not None:
            reward = self.reward_fn(state=state, action=action, new_state=new_state)

        return action, new_state, reward, done, info

    def score_policy(self, gamma, epsilon, render=False):
        """
        To track learning progress, periodically score the policy
        This function rolls out the policy and returns the episode's return

        :param gamma: float, the discount factor
        :param epsilon: float, the e-greedy action selection factor
        :return: (episode_sum_rewards, episode_return)
            episode_sum_rewards - the total reward over the rollout; not discounted
            episode_return - discounted reward over the rollout
            episode_fitness - if a fitness (aka true reward) function is defined, the fitness count
        """
        state = self.env.reset()

        episode_sum_rewards = 0
        episode_return = 0
        episode_fitness = 0
        j = 0
        done = False

        while not done:
            if render:
                self.env.render()
                time.sleep(0.1)
            if "hungry-thirsty" in self.env.spec.id and not state["hungry"]:
                episode_fitness += 1

            action, state, reward, done, info = self.e_greedy_action_selection(state=state, epsilon=epsilon)
            episode_sum_rewards += reward
            episode_return += (gamma ** j) * reward
            j += 1

        return episode_sum_rewards, episode_return, episode_fitness

    def record_performance(self, episode, gamma, epsilon, num_tests, record_type='return'):
        """
        Record the performance of the policy

        :param episode: int (corresponding to the training episode)
        :param num_tests: int (number of test comparisons for variance reduction)
        :param record_type: 'return' or 'sum_scores'
        :return:
        """
        scores = [episode, []]
        for idx in range(num_tests):
            episode_sum_rewards, episode_return, episode_fitness = self.score_policy(gamma=gamma, epsilon=epsilon)
            if record_type == 'return':
                scores[1].append(episode_return)
            elif record_type == 'sum_scores':
                scores[1].append(episode_sum_rewards)
            elif record_type == 'fitness':
                scores[1].append(episode_fitness)
            else:
                raise Exception("record_type failure (not 'return' or 'sum_scores')")
        self.learning_performance.append(scores)

    def learn_1_episode(self, alpha_lr, gamma, epsilon, record=True):
        """

        :param alpha_lr:
        :param gamma:
        :param epsilon:
        :param record:
        :return:
        """
        # reset for learning
        state = self.env.reset()
        done = False
        fitness = 0
        step = 0
        episode_fitness = []

        while not done:
            step += 1
            if not state["hungry"]:
                fitness += 1

            action, new_state, reward, done, info = self.e_greedy_action_selection(state=state, epsilon=epsilon)

            # update the Q table
            best_next_action = self.q_table_argmax(new_state)
            new_state_q_val = self.q_lookup(state=new_state, action=best_next_action)
            update_value = alpha_lr * (reward + gamma * new_state_q_val)
            self.q_update(state=state, action=action, update_value=update_value, alpha_lr=alpha_lr)

            # update the state
            state = new_state
            if record:
                episode_fitness.append([step, [fitness]])

            if done:
                return episode_fitness

    def get_action(self, state):
        """
        Select an action using e-greedy based on the state & current q table

        :param state: a state
        :return: action (int)
        """
        # use epsilon-greedy action selection
        if random.random() < self.epsilon:
            action = self.env.get_random_action()
        else:
            action = self.q_table_argmax(state)
        return action

    def learn_n_episodes(self, alpha_lr, gamma, record_freq, epsilon, num_tests):
        """
        Update Q table and record performance over time

        :param alpha_lr: float, the learning rate
        :param gamma: float, the discount factor
        :param record_freq: int, the frequency with which to assess performance
        :param epsilon: float, the e-greedy factor
        :param num_tests: int, the number of trials to run
        :return: None. Q_table is a class variable, updated in situ
        """
        for i in tqdm(range(self.num_episodes)):
            self.learn_1_episode(alpha_lr=alpha_lr, gamma=gamma, epsilon=epsilon, record=False)
            # record learning performance every RECORD episodes
            if i % record_freq == 0:
                self.record_performance(episode=i,
                                        gamma=gamma,
                                        epsilon=epsilon,
                                        num_tests=num_tests,
                                        record_type=self.record_type)
        return self.learning_performance


def create_q_learning_agent(env, hyper_params, user_reward_fn):
    """
    Create a q-learning agent and return its learning performance

    :param env: An openai gym environment
    :param hyper_params: a dict. Must contain alpha_lr (float), gamma (float), epsilon (float), num_episodes (int)
                            num_tests (int), record_freq (int), and num_episodes (int)
    :param user_reward_fn: a user-defined reward fn f(s,a,s') -> R
    :return: alg (the agent), alg.learning_performance (a list corresponding to the agent's learning performance)
    """
    assert ("alpha_lr" in hyper_params.keys())
    assert ("gamma" in hyper_params.keys())
    assert ("epsilon" in hyper_params.keys())
    assert ("num_episodes" in hyper_params.keys())
    assert ("num_tests" in hyper_params.keys())
    assert ("record_freq" in hyper_params.keys())
    assert ("num_episodes" in hyper_params.keys())

    """ Q-Learning hyperparameters """
    alpha_lr = hyper_params["alpha_lr"]
    epsilon = hyper_params["epsilon"]
    gamma = hyper_params["gamma"]
    num_tests = hyper_params["num_tests"]
    record_freq = hyper_params["record_freq"]
    num_episodes = hyper_params["num_episodes"]

    alg = QLearning(env, num_episodes=num_episodes, reward_fn=user_reward_fn, epsilon=epsilon)
    if num_episodes == 1:
        return alg, alg.learn_1_episode(alpha_lr=alpha_lr,
                                        gamma=gamma,
                                        epsilon=epsilon)
    else:
        alg.learn_n_episodes(alpha_lr=alpha_lr,
                             gamma=gamma,
                             record_freq=record_freq,
                             epsilon=epsilon,
                             num_tests=num_tests)
        return alg, alg.learning_performance


if __name__ == "__main__":
    env_name = 'hungry-thirsty-v0'
    env = gym.make(env_name)
    if 'hungry-thirsty' in env_name:
        env.update_step_limit(200)
    env.reset()

    """ Q-Learning hyperparameters """
    alpha_lr = 0.05
    epsilon = 0.2
    gamma = 0.99
    num_tests = 1
    record_freq = 10
    num_episodes = 10000

    if num_episodes > 1:
        alg = QLearning(env, num_episodes=num_episodes, reward_fn=wdrcf_reward_fn)
        alg.learn_n_episodes(alpha_lr=alpha_lr,
                             gamma=gamma,
                             record_freq=record_freq,
                             epsilon=epsilon,
                             num_tests=num_tests)
        plot_learning_curve(alg.learning_performance)
    else:
        alg = QLearning(env, num_episodes=num_episodes, reward_fn=sparse_reward_fn)
        perf = alg.learn_1_episode(alpha_lr, gamma, epsilon, record=True)
        plot_learning_curve(perf)

    # show agent
    while True:
        print("Would you like to see the agent? Y/N")
        user_input = input()
        if user_input == "y" or user_input == "Y":
            alg.score_policy(gamma=gamma, epsilon=epsilon, render=True)
            alg.env.close()
        else:
            break

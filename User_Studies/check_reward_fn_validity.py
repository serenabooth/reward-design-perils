import csv
import gym
import sys, os
sys.path.insert(0, 'RL_algorithms')
import Value_Iteration
from gym_hungry_thirsty.envs.hungry_thirsty_reward_fns import *
from gym_hungry_thirsty.envs.hungry_thirsty_env import compute_reward
from Utils import find_filenames_with_extension

adversarial_choices = [True, False]


"""
For each expert-specified reward function, check its validity
by comparing the resulting policy (learned with VI) to the 
policy (also learned with VI) 
"""
BASE_DIR = "User_Studies/Expert-User-Study/user_tests/"

theta = 0.01
num_test_episodes = 100
optimal_performance = {}

for adversarial in adversarial_choices:
    env_6x6 = gym.make('hungry-thirsty-v0', size=(6, 6))
    env_4x4 = gym.make('hungry-thirsty-v0', size=(4, 4))
    if adversarial:
        env_6x6.reset(food_loc=(0, 0), water_loc=(0, 5))
        env_4x4.reset(food_loc=(0, 0), water_loc=(0, 3))
        CSV_BASE_NAME = 'viability_of_fns_adv.csv'
    else:
        env_6x6.reset(food_loc=(5, 0), water_loc=(5, 5))
        env_4x4.reset(food_loc=(3, 0), water_loc=(3, 3))
        CSV_BASE_NAME = 'viability_of_fns_non_adv.csv'

    vi_alg_6x6 = Value_Iteration.Value_Iteration(env_6x6, reward_fn=sparse_reward_fn)
    vi_alg_4x4 = Value_Iteration.Value_Iteration(env_4x4, reward_fn=sparse_reward_fn)

    optimal_performance_6x6 = {}
    optimal_performance_4x4 = {}

    for user in range(0, 30):
        # 6x6 setting (set once!)
        if user < 12:
            CSV_NAME = "6x6_" + CSV_BASE_NAME
            env = env_6x6
            optimal_performance = optimal_performance_6x6
        # 4x4 setting (set once!)
        elif user < 30:
            CSV_NAME = "4x4_" + CSV_BASE_NAME
            env = env_4x4
            optimal_performance = optimal_performance_4x4
        else:
            raise Exception("User ID does not exist; something has gone wrong!")

        file_exists = os.path.isfile(BASE_DIR + CSV_NAME)

        with open(BASE_DIR + CSV_NAME, 'a', newline='') as record_fn_file:
            record_fn_file_writer = csv.writer(record_fn_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if not file_exists:
                record_fn_file_writer.writerow(["User",
                                                "Gamma",
                                                "Reward Fn",
                                                "Viable?",
                                                "Performance",
                                                "Comparison"])

            csvname = find_filenames_with_extension(BASE_DIR + str(user), extension=".csv")[0]
            print("User: ", user)
            with open(BASE_DIR + str(user) + "/" + csvname, newline='') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
                header = None
                for row in csvreader:
                    if header is None:
                        header = row
                        continue
                    reward_fn_dict = eval(row[2])
                    hyper_params = eval(row[3])

                    reward_scaling_factor = hyper_params["reward_scaling_factor"]
                    gamma = hyper_params['gamma']
                    if gamma > 0.99:
                        gamma = 0.999
                    for key in reward_fn_dict:
                        reward_fn_dict[key] *= reward_scaling_factor


                    def reward_fn(state, action, new_state):
                        """
                        wrapper for compute_reward function, which instantiates reward params

                        :param state:
                        :param action:
                        :param new_state:
                        :return:
                        """
                        return compute_reward(reward_fn_params=reward_fn_dict, state=state)


                    if gamma not in optimal_performance.keys():
                        alg = Value_Iteration.Value_Iteration(env, reward_fn=sparse_reward_fn)
                        alg.value_iteration_training(theta=theta, gamma=gamma)
                        alg.policy_training(gamma=gamma)
                        optimal_performance[gamma] = alg.compute_avg_fitness(num_tests=num_test_episodes,
                                                                             random_seed=0)

                    optimal_policy_avg = optimal_performance[gamma]
                    print("Comparing performance to: ", optimal_policy_avg, " with gamma: ", gamma)
                    alg = Value_Iteration.Value_Iteration(env, reward_fn=reward_fn)
                    alg.value_iteration_training(theta=theta, gamma=gamma)
                    alg.policy_training(gamma=gamma)
                    avg_perf = alg.compute_avg_fitness(num_tests=num_test_episodes,
                                                       random_seed=0)
                    if abs(avg_perf - optimal_policy_avg) < 0.05:
                        print(avg_perf)
                    else:
                        print("Discarded reward fn: ", reward_fn_dict, " with performance: ", avg_perf)
                    record_fn_file_writer.writerow([user,
                                                    gamma,
                                                    reward_fn_dict,
                                                    bool(abs(avg_perf - optimal_policy_avg) < 0.05),
                                                    avg_perf,
                                                    optimal_policy_avg])

            print("\n\n")

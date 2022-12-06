from itertools import product
import eval_reward_fn
import numpy as np
import csv
import gym
import os
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, 'RL_algorithms')
from Utils import tsplot, find_filenames_with_extension
from default_parameters import default_param_lookup
import argparse

"""
This file is the backbone for computing true cumulative performance for each configuration.
"""

gym.logger.set_level(40)
BASE_DIR = "Experiments/saved_reward_fn_performances/"


def main(search_type, alg, hyperparameter, hyperparameter_values):
    """

    :param search_type: string
    :param alg: string
    :param hyperparameter: string
    :param hyperparameter_values: list of floats
    :return:
    """
    reward_fns = []
    num_environments = None

    if search_type == "Large":
        """
        Do a large reward fn search - 5091 reward fns 
        """
        selected_hyper_params = default_param_lookup[alg].copy()
        local_dir = BASE_DIR + "{}".format(str(selected_hyper_params))

        if os.path.exists(local_dir):
            completed_reward_functions = find_filenames_with_extension(local_dir, "csv")
            completed_reward_functions = [fn.replace(".csv", "").replace("csv", "").replace("reward_fn_perf_", "")
                                          for fn in completed_reward_functions]
            completed_reward_functions = [eval(fn) for fn in completed_reward_functions]
        else:
            completed_reward_functions = []

        x = [-1.0, -0.5, -0.1, -0.05, 0, 0.05, 0.1, 0.5, 1.0]
        candidate_reward_fns = list(product(x, x, x, x))

        # reward functions must be roughly logically consistent
        for item in candidate_reward_fns:
            # don't add reward functions which are already finished
            if item in completed_reward_functions:
                pass
            # don't add invalid reward functions
            elif not (item[0] >= item[2] and item[0] >= item[3] and item[1] >= item[2] and item[1] >= item[3]):
                reward_fns.append(item)

    elif search_type == "User_All":
        """
        Do a search with ALL user-generated reward functions (including invalid ones)
        """
        with open('User_Studies/Expert-User-Study/user_tests/all_reward_fns.csv', newline='') as csvf:
            csvreader = csv.reader(csvf, delimiter=',', quotechar='"')
            for idx, row in enumerate(csvreader):
                if idx == 0:
                    continue
                reward_fns.append(eval(row[2]))

    elif search_type == "User_Unique":
        """
        Do a search with only the unique user-generated reward functions (only non-adversarially-valid ones)
        """
        with open('User_Studies/Expert-User-Study/user_tests/unique_reward_fns.csv', newline='') as csvf:
            csvreader = csv.reader(csvf, delimiter=',', quotechar='"')
            for idx, row in enumerate(csvreader):
                if idx == 0:
                    continue
                elif row[2] == "TRUE":  # only add valid (non-adv) reward fns
                    reward_fns.append(eval(row[0]))
        print(len(reward_fns))

    elif search_type == "Deep_Dive":
        """
        Do a 'deep dive' to average across more tests to minimize error stemming from stochasticity in env or training
        """
        if alg == "Q_Learn":
            num_environments = 1000
        else:
            num_environments = 30

        combination = []
        top_n = 3
        param = hyperparameter
        hyper_param_choices = hyperparameter_values
        assert (len(hyper_param_choices) == 2)
        for val in hyper_param_choices:
            if val in ['A2C', 'DDQN', 'PPO']:
                combination.append(val)
            elif param == "alpha_lr" and val == 0.05:
                combination.append("Baseline")
            elif param == "alpha_lr":
                combination.append("{}={}".format("Alpha", val))
            elif param == "gamma" and val == 0.99:
                combination.append("Baseline")
            elif param == "gamma":
                combination.append("{}={}".format("Gamma", val))
            else:
                raise Exception("Incorrect parameter specification.")

        combination = tuple(combination)
        with open(BASE_DIR + 'AAAI_Experiments/TopDiffs/top_reward_fn_diffs_{}'.format(combination),
                  newline="") as csvf:
            csvreader = csv.reader(csvf, delimiter=',', quotechar='"')
            for idx, row in enumerate(csvreader):
                if idx == 0 or idx == 1:
                    continue
                elif idx <= top_n:
                    if type(eval(eval(row[2])[0])) is tuple:
                        reward_fns.append([float(eval(val)[1]) for val in eval(row[2])])
                    else:
                        reward_fns.append([float(val) for val in eval(row[2])])



        reward_fns_tmp = []
        for combination_entry in combination:
            with open(BASE_DIR + 'AAAI_Experiments/TopOverallFns/top_reward_fn_{}'.format(combination_entry),
                      newline="") as csvf:
                csvreader = csv.reader(csvf, delimiter=',', quotechar='"')
                for idx, row in enumerate(csvreader):
                    if idx == 0:
                        continue
                    elif idx <= top_n:
                        if type(eval(eval(row[2])[0])) is tuple:
                            reward_fn_candidate = [float(eval(val)[1]) for val in eval(row[2])]
                        else:
                            reward_fn_candidate = [float(val) for val in eval(row[2])]
                        if reward_fn_candidate not in reward_fns:
                            reward_fns_tmp.append(reward_fn_candidate)
        reward_fns = reward_fns + reward_fns_tmp
    else:
        print("search_type set incorrectly; exiting")
        exit()

    print("num reward fns: ", len(reward_fns))
    fitness_plotting = []
    all_reward_fn_performances = {}

    for value in hyperparameter_values:
        selected_hyper_params = default_param_lookup[alg].copy()
        selected_hyper_params[hyperparameter] = value
        if num_environments is not None:
            selected_hyper_params["num_environments"] = num_environments

        local_dir = BASE_DIR + "{}/".format(search_type)
        if not os.path.exists(local_dir):
            os.mkdir(local_dir)
        local_dir = BASE_DIR + "{}/{}/".format(search_type, alg)
        if not os.path.exists(local_dir):
            os.mkdir(local_dir)

        local_dir = BASE_DIR + "{}/{}/{}".format(search_type, alg, str(selected_hyper_params))
        local_dir = local_dir.replace("neural_net_hidden_size", "nn_size")
        local_dir = local_dir.replace("plotting_steps", "plt")
        local_dir = local_dir.replace("num_environments", "envs")
        local_dir = local_dir.replace("epsilon", "eps")
        local_dir = local_dir.replace("exp_replay_size", "rply_sz")
        local_dir = local_dir.replace("sync_frequency", "sync_frq")

        if not os.path.isdir(local_dir):
            os.mkdir(local_dir)
        else:
            print("Dir: ", local_dir, "already exists")

        existing_files = find_filenames_with_extension(local_dir, ".csv")

        def eval_params(reward_fn_params):
            """
            Evaluate each set of reward function parameters

            :param reward_fn_params:
            :return:
            """
            file_extension = "reward_fn_perf_{}.csv".format(str(reward_fn_params))
            # print (file_extension)
            if file_extension in existing_files:
                return

            filename = local_dir + "/" + file_extension
            print("Reward fn: ", reward_fn_params, "\n\n")
            fitness_all_episodes = eval_reward_fn.eval_reward_fn(alg=selected_hyper_params["alg"],
                                                                 reward_fn_params=reward_fn_params,
                                                                 hyper_params=selected_hyper_params,
                                                                 num_trials=selected_hyper_params["num_environments"])
            print("Reward Fn Assessment Done")
            print("\n\n")

            with open(filename, "w+", newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                # writing the fields
                hyperparams = []
                for x in selected_hyper_params.items():
                    hyperparams.append(x)
                csvwriter.writerow(hyperparams)
                if type(reward_fn_params) is dict:
                    csvwriter.writerow(list(reward_fn_params.items()))
                else:
                    csvwriter.writerow(reward_fn_params)

                for datum in fitness_all_episodes:
                    csvwriter.writerow(datum)

            f, ax = plt.subplots()
            data = np.array(fitness_all_episodes)
            if len(data.shape) == 3:
                data = np.concatenate(data, axis=-1)
            tsplot(ax, data=data)
            plt.ylabel("Return")
            plt.xlabel("Episode")
            filename_pdf = local_dir + "/reward_fn_perf_{}.pdf".format(str(reward_fn_params))
            plt.savefig(filename_pdf)

        for reward_fn in reward_fns:
            eval_params(reward_fn_params=reward_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--search_type',
                        help='choice of: "Large" | "User_All" | "User_Unique" | "Deep_Dive"',
                        required=True)
    parser.add_argument('--test_alg',
                        help='choice of: "Q_Learn" | "A2C" | "DDQN" | "PPO"',
                        required=True)
    parser.add_argument('--hyperparameter',
                        help='choice of: "gamma" | "alpha_lr" | "alg"',
                        required=True)
    parser.add_argument('--hyperparameter_values', '--list', action='append',
                        help='list of numbers, e.g., ',
                        required=True)
    args = parser.parse_args()
    search_type = args.search_type
    alg = args.test_alg
    hyperparameter = args.hyperparameter
    if hyperparameter in ["gamma", "alpha_lr"]:
        hyperparameter_values = [float(val) for val in args.hyperparameter_values]
        hyperparameter_values = sorted(hyperparameter_values, reverse=True)
    else:
        hyperparameter_values = args.hyperparameter_values
    assert (search_type in ["Large", "User_All", "User_Unique", "Deep_Dive"])
    assert (alg in ["Q_Learn", "A2C", "DDQN", "PPO"])
    assert (hyperparameter in ["gamma", "alpha_lr", "alg"])
    main(search_type, alg, hyperparameter, hyperparameter_values)

import csv
import numpy as np
import math
import scipy.stats

import sys
sys.path.insert(0, 'RL_algorithms')
from default_parameters import default_param_lookup
import argparse

"""
Compute the Hoeffding Bound & Mann Whitney U-test value 
"""


BASE_DIR = "Experiments/saved_reward_fn_performances/AAAI_Experiments/"


def read_csv_into_memory(csv_filename):
    """
    Read csv

    :param csv_filename: string
    :return: np array
    """
    csv_file = open(csv_filename, "r")
    csv_reader = csv.reader(csv_file, delimiter=',')
    hyper_params = next(csv_reader)  # gets the first line (which just records hyperparams)
    reward_fn = next(csv_reader)  # gets the second line (which just records the reward fn)
    performance = []
    for idx, row in enumerate(csv_reader):
        running_total = 0
        row = [[running_total := running_total + eval(x)[0] - 1] for x in row]  # subtract 1 since every ep starts -H
        performance.append(row[-1])

    data = np.array(performance)
    return data


def compute_hoeffding_bound(data, delta=0.1, upper_bound=150000):
    """
    Compute the hoeffding bound - https://people.cs.umass.edu/~pthomas/papers/Thomas2015.pdf

    :param data: np array
    :param delta: float, the probability is bounded by 1-delta
    :param upper_bound: upper bound on values in the data
    :return: mean, hoeffding_bnd_lower, hoeffding_bnd_upper
    """
    data_mean = np.mean(data)
    hoeffding_bnd_lower = data_mean - upper_bound * math.sqrt(math.log(1 / delta) / (2 * len(data)))
    hoeffding_bnd_upper = data_mean + 2 * upper_bound * math.sqrt(math.log(1 / delta) / (2 * len(data)))

    return data_mean, hoeffding_bnd_lower, hoeffding_bnd_upper


def get_filename(alg, reward_fn_params, num_envs, search_type="Deep_Dive", lookup_key=None, lookup_value=None):
    """
    Return the filename (having made all the hyperparam changes necessary for the lookup)

    :param alg: String, the algorithm
    :param reward_fn_params: list
    :param num_envs: int
    :param lookup_key:
    :param lookup_value:
    :return: String, the filename
    """
    selected_hyper_params = default_param_lookup[alg].copy()
    if lookup_key is not None:
        selected_hyper_params[lookup_key] = lookup_value
    selected_hyper_params["num_environments"] = num_envs
    local_dir = BASE_DIR + "{}/{}/{}".format(search_type, alg, str(selected_hyper_params))
    local_dir = local_dir.replace("neural_net_hidden_size", "nn_size")
    local_dir = local_dir.replace("plotting_steps", "plt")
    local_dir = local_dir.replace("num_environments", "envs")
    local_dir = local_dir.replace("epsilon", "eps")
    local_dir = local_dir.replace("exp_replay_size", "rply_sz")
    local_dir = local_dir.replace("sync_frequency", "sync_frq")
    filename_pdf = local_dir + "/reward_fn_perf_{}.csv".format(str(reward_fn_params))
    return filename_pdf


def main(study_type, hyperparameter_or_alg, hyperparameter_or_alg_values, num_comparisons):
    """

    :param study_type: string, the type of study
    :param hyperparameter_or_alg: string, one of "alg", "gamma", or "alpha_lr"
    :param hyperparameter_or_alg_values: list, the list of values the hyperparameter or alg can take on
    :param num_comparisons: int, the number of reward functions to assess
    :return:
    """
    reverse_order = False
    if hyperparameter_or_alg != "alg":
        alg = "Q_Learn"
        num_envs = 1000
    else:
        num_envs = 30

    combination = []

    for val in hyperparameter_or_alg_values:
        if hyperparameter_or_alg == "alpha_lr" and val == 0.05:
            combination.append("Baseline")
        elif hyperparameter_or_alg == "alpha_lr":
            combination.append("{}={}".format("Alpha", val))
        elif hyperparameter_or_alg == "gamma" and val == 0.99:
            combination.append("Baseline")
        elif hyperparameter_or_alg == "gamma":
            combination.append("{}={}".format("Gamma", val))
        else:
            combination.append(val)

    reward_fns = {}
    combination = tuple(combination)
    if study_type == "TOP_DIFF_H1":
        reward_fns["diffs"] = {}
        with open(BASE_DIR + 'TopDiffs/top_reward_fn_diffs_{}'.format(combination), newline="") as csvf:
            csvreader = csv.reader(csvf, delimiter=',', quotechar='"')
            for idx, row in enumerate(csvreader):
                if idx == 0:
                    continue
                elif idx <= num_comparisons:
                    reward_fn_candidate = eval(row[2])
                    if type(eval(reward_fn_candidate[0])) is tuple:
                        reward_fn_candidate = [float(eval(val)[1]) for val in reward_fn_candidate]
                    else:
                        reward_fn_candidate = [float(val) for val in reward_fn_candidate]
                    if combination in reward_fns["diffs"].keys():
                        reward_fns["diffs"][combination].append(reward_fn_candidate)
                    else:
                        reward_fns["diffs"][combination] = [reward_fn_candidate]

    if study_type == "OPTIMALS_H2":
        reward_fns["optimals"] = {}
        for combination_entry in combination:
            with open(BASE_DIR + 'TopOverallFns/top_reward_fn_{}'.format(combination_entry),
                      newline="") as csvf:
                csvreader = csv.reader(csvf, delimiter=',', quotechar='"')
                for idx, row in enumerate(csvreader):
                    if idx == 0:
                        continue
                    elif idx <= num_comparisons:
                        reward_fn_candidate = eval(row[2])
                        if type(eval(reward_fn_candidate[0])) is tuple:
                            reward_fn_candidate = [float(eval(val)[1]) for val in reward_fn_candidate]
                        else:
                            reward_fn_candidate = [float(val) for val in reward_fn_candidate]
                        if "{}".format(combination_entry) in reward_fns["optimals"].keys():
                            reward_fns["optimals"]["{}".format(combination_entry)].append(reward_fn_candidate)
                        else:
                            reward_fns["optimals"]["{}".format(combination_entry)] = [reward_fn_candidate]

    # compute statistics for max diff reward functions
    if "diffs" in reward_fns.keys():
        print ("Maximally different reward functions: ")
        for reward_fn_params in reward_fns['diffs'][combination]:
            all_data = {}
            for value in hyperparameter_or_alg_values:
                if hyperparameter_or_alg == "alg":
                    alg = value
                    lookup_key = None
                    lookup_value = None
                else:
                    lookup_key = hyperparameter_or_alg
                    lookup_value = value

                    filename_pdf = get_filename(alg=alg,
                                                lookup_key=lookup_key,
                                                num_envs=num_envs,
                                                lookup_value=lookup_value,
                                                reward_fn_params=reward_fn_params)

                    all_data["{}.{}.{}".format(reward_fn_params,
                                               hyperparameter_or_alg,
                                               value)] = read_csv_into_memory(filename_pdf)
                    mean, hl, hu = compute_hoeffding_bound(all_data["{}.{}.{}".format(reward_fn_params,
                                                                                      hyperparameter_or_alg,
                                                                                      value)])
                    print ("{} & $\{}={}$ & [{}, {}]".format(reward_fn_params,
                                                             hyperparameter_or_alg,
                                                             value,
                                                             int(hl),
                                                             int(hu)))

            dict_keys = list(all_data.keys())
            # print ("Mann Whitney for {} and {}".format(dict_keys[0], dict_keys[1]))
            _, p_value = scipy.stats.mannwhitneyu(x=all_data[dict_keys[0]].flatten(), y=all_data[dict_keys[1]].flatten())
            print ("{} Mann Whitney p_value: {}".format(reward_fn_params, p_value), "\n")

    # compute statistics for optimal reward functions
    if "optimals" in reward_fns.keys():
        assert (len(combination) == 2)
        key = hyperparameter_or_alg
        assessment_value = hyperparameter_or_alg_values[0]

        for idx, optimal_reward_fn in enumerate(reward_fns['optimals']["{}".format(combination[0])]):
            all_data = {}
            print ("\n\n\n\n", idx, " : top reward function, ", optimal_reward_fn, " for ", combination[0])
            filename_pdf = get_filename(alg=alg,
                                        num_envs=num_envs,
                                        lookup_key=key,
                                        lookup_value=assessment_value,
                                        reward_fn_params=optimal_reward_fn)
            optimal_reward_fn_data = read_csv_into_memory(filename_pdf)
            mean, hl, hu = compute_hoeffding_bound(optimal_reward_fn_data)
            print("Mean: {} , HL/HU: [{},{}]".format(int(mean), int(hl), int(hu)))
            for suboptimal_idx, alternative_reward_fn in enumerate(reward_fns['optimals']["{}".format(combination[1])]):
                print ("Alternative reward function {} for {}".format(suboptimal_idx, combination[1]))
                print("{}: {}... {}".format(key, assessment_value, alternative_reward_fn))
                filename_pdf = get_filename(alg=alg,
                                            num_envs=num_envs,
                                            lookup_key=key,
                                            lookup_value=assessment_value,
                                            reward_fn_params=alternative_reward_fn)
                suboptimal_reward_fn_data = read_csv_into_memory(filename_pdf)
                mean, hl, hu = compute_hoeffding_bound(suboptimal_reward_fn_data)
                print ("Mean: {} , HL/HU: [{},{}]".format(int(mean), int(hl), int(hu)))
                print("Greater Mann Whitney :", scipy.stats.mannwhitneyu(x=optimal_reward_fn_data.flatten(),
                                                                         y=suboptimal_reward_fn_data.flatten(),
                                                                         alternative='greater'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--study_type',
                        help='choice of: "TOP_DIFF_H1" | "OPTIMALS_H2"',
                        required=True)
    parser.add_argument('--hyperparameter_or_alg',
                        help='choice of: "gamma" | "alpha_lr" | "alg"',
                        required=True)
    parser.add_argument('--hyperparameter_or_alg_values', '--list', action='append',
                        help='list of values',
                        required=True)
    parser.add_argument('--num_comparisons', default=1, type=int)
    args = parser.parse_args()
    study_type = args.study_type
    hyperparameter_or_alg = args.hyperparameter_or_alg
    hyperparameter_or_alg_values = args.hyperparameter_or_alg_values

    assert (study_type in ["TOP_DIFF_H1", "OPTIMALS_H2"])
    assert (hyperparameter_or_alg in [ "gamma", "alpha_lr", "alg"])
    if hyperparameter_or_alg in ["gamma", "alpha_lr"]:
        hyperparameter_or_alg_values = [float(val) for val in hyperparameter_or_alg_values]

    main(study_type=study_type,
         hyperparameter_or_alg=hyperparameter_or_alg,
         hyperparameter_or_alg_values=hyperparameter_or_alg_values,
         num_comparisons=args.num_comparisons)
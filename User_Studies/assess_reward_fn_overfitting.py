import sys

sys.path.insert(0, 'Experiments')
sys.path.insert(0, 'RL_algorithms')
import eval_reward_fn
from Utils import tsplot, find_filenames_with_extension
import csv
import matplotlib.pyplot as plt
import numpy as np
ALL_USER_TESTS = "User_Studies/Expert-User-Study/user_tests/all_reward_fns.csv"

"""
Re-compute each user's cumulative performance (mostly for distributing bonuses) 
"""

def main():
    overfitting_experiments = find_filenames_with_extension('Experiments/saved_reward_fn_performances/'
                                                            'AAAI_Experiments/UserPerformance',
                                                            ".pdf")
    overfitting_experiments_indeces = [int(datum.replace('.pdf','')) for datum in overfitting_experiments]
    print ("Number of experiments done: ", len(overfitting_experiments_indeces))
    with open(ALL_USER_TESTS, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        i = 0
        for idx, row in list(enumerate(csvreader)):
            # print (idx)
            assert len(row) == 8, "Row must have 8 entries"
            if idx == 0 or idx == 143:
                continue
            if idx in overfitting_experiments_indeces:
                continue
            else:
                pass


            # if idx > 250:
            #     exit()

            user_id = eval(row[0])
            gamma = eval(row[1])
            reward_fn_params = eval(row[2])
            valid_adv = eval(row[3])
            valid_non_adv = eval(row[4])
            if valid_non_adv:
                print (idx)
            else:
                continue
            i += 1

            alg_choice = row[5]
            hyper_params = eval(row[6])
            selected_agent = eval(row[7])

            # fill in hyper-param values user had no control over
            hyper_params["max_steps"] = 200
            hyper_params["sync_frequency"] = 5
            hyper_params["plot_update_freq"] = 100
            hyper_params["neural_net_hidden_size"] = 144
            hyper_params["neural_net_extra_layers"] = 0
            hyper_params["exp_replay_size"] = 5000
            hyper_params["K_epochs"] = 80
            hyper_params["n_step_update"] = 20

            env_size = (6, 6)
            if user_id >= 12:
                env_size = (4, 4)

            # hyper_params["num_episodes"] = 10

            fitness_all_episodes = eval_reward_fn.eval_reward_fn(alg=alg_choice,
                                                                 hyper_params=hyper_params,
                                                                 reward_fn_params=reward_fn_params,
                                                                 env_name='hungry-thirsty-v0',
                                                                 num_trials=10,
                                                                 env_size=env_size)

            with open("Experiments/saved_reward_fn_performances/"
                      "AAAI_Experiments/UserPerformance/overfitting_checks/%d.csv" % idx, 'a', newline='') as saved_perf_csv:
                csv_writer = csv.writer(saved_perf_csv, delimiter=',', quotechar='"')
                csv_writer.writerow(row)
                for datum in fitness_all_episodes:
                    csv_writer.writerow(datum)

            f, ax = plt.subplots()
            data = np.array(fitness_all_episodes)
            if len(data.shape) == 3:
                data = np.concatenate(data, axis=-1)
            tsplot(ax, data=data)
            plt.ylabel("Return")
            plt.xlabel("Episode")
            plt.savefig("Experiments/saved_reward_fn_performances/"
                        "AAAI_Experiments/UserPerformance/overfitting_checks/%d.pdf" % idx)


if __name__ == "__main__":
    main()

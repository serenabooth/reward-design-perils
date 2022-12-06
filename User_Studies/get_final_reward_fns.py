import csv
import sys
sys.path.insert(0, 'RL_algorithms')
from gym_hungry_thirsty.envs.hungry_thirsty_reward_fns import *
from Utils import find_filenames_with_extension


"""
Write a CSV of all user study data.
Save to either 'all_reward_fns.csv' or 'final_reward_fns.csv'
 
if FINAL_ONLY, only record the user's submitted configuration 
"""
FINAL_ONLY = False


BASE_DIR = "User_Studies/Expert-User-Study/user_tests/"

if FINAL_ONLY:
    CSV_NAME = "final_reward_fns.csv"
else:
    CSV_NAME = "all_reward_fns.csv"

with open(BASE_DIR + CSV_NAME, 'w', newline='') as record_fn_file:
    record_fn_file_writer = csv.writer(record_fn_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    record_fn_file_writer.writerow(["User",
                                    "Gamma",
                                    "Reward Fn",
                                    "Valid (adversarial)?",
                                    "Valid (not adversarial)?",
                                    "Alg",
                                    "Params",
                                    "Selected Agent?"])

    data_entry_idx = 1
    reset_csv_id = True
    for user in range(0, 30):
        txtname = find_filenames_with_extension(BASE_DIR + str(user), ".txt")[0]
        with open(BASE_DIR + str(user) + "/" + txtname) as f:
            selected_agent = f.readline()
            selected_agent = int(selected_agent.replace("Selected agent: ", ""))

        csv_id = "6x6"
        if user >= 12:
            csv_id = "4x4"
            if reset_csv_id:
                reset_csv_id = False
                data_entry_idx = 1


        csvname = find_filenames_with_extension(BASE_DIR + str(user), ".csv")[0]
        with open(BASE_DIR + str(user) + "/" + csvname, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for idx, row in enumerate(csvreader):
                if idx == 0:
                    continue
                if FINAL_ONLY and idx != selected_agent + 1:
                    data_entry_idx += 1
                    continue

                with open(BASE_DIR + csv_id + "_viability_of_fns_adv.csv", newline='') as viability_adv_f:
                    csvreader_viability_adv = csv.reader(viability_adv_f, delimiter=',', quotechar='"')
                    for viability_idx, viability_row in enumerate(csvreader_viability_adv):
                        if viability_idx == 0:
                            continue
                        if viability_idx != data_entry_idx:
                            continue
                        else:
                            validity_adv = viability_row[3]

                with open(BASE_DIR + csv_id + "_viability_of_fns_non_adv.csv", newline='') as viability_non_adv_f:
                    csvreader_viability_non_adv_f = csv.reader(viability_non_adv_f, delimiter=',', quotechar='"')
                    for viability_idx, viability_row in enumerate(csvreader_viability_non_adv_f):
                        if viability_idx == 0:
                            continue
                        if viability_idx != data_entry_idx:
                            continue
                        else:
                            validity_non_adv = viability_row[3]

                alg = row[1]
                reward_fn_dict = eval(row[2])
                hyper_params = eval(row[3])
                gamma = hyper_params['gamma']
                reward_scaling_factor = hyper_params["reward_scaling_factor"]
                for key in reward_fn_dict:
                    reward_fn_dict[key] *= reward_scaling_factor
                record_fn_file_writer.writerow([user,
                                                gamma,
                                                reward_fn_dict,
                                                validity_adv,
                                                validity_non_adv,
                                                alg,
                                                hyper_params,
                                                idx == selected_agent+1])
                data_entry_idx += 1


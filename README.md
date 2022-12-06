# AAAI 2023: The Perils of Trial-and-Error Reward Design: Misdesign through Overfitting and Invalid Task Specifications

Serena Booth, W. Bradley Knox, Julie Shah, Scott Niekum, Peter Stone, and Alessandro Allievi 

Bosch, The University of Texas at Austin, MIT CSAIL, The University of Massachusetts Amherst, Google Research, and Sony AI 

Contact author: sbooth@mit.edu 

## Codebase Structure

This work is divided into the following directories:
* ```Domains```
  * This contains the Hungry-Thirsty Domain code, which is written as an OpenAI Gym Environment.
  * Files of interest: ```Domains/gym-hungry-thirsty/gym_hungry_thirsty/envs/hungry_thirsty_env.py``` is the main implementation.
  * ```Domains/hungry-thristy-user-control.py``` lets you control the agent directly, which gives a feel for the task. 
  
* ```Experiments```
  * This directory contains code for running all of the computational experiments, as well as for analyzing the user study data. 
  * All data is shared and saved in ```Experiments/saved_reward_fn_performances/AAAI_Experiments/*```.
  * The main file for running computational studies is ```reward_fn_performance_assessments.py```. This script trains RL agents with different reward functions N times and records their cumulative true performance. 
  * Experiments/plotting contains two Jupyter Notebooks. 
    * ```parallel_coord_ipynb.ipynb``` creates csv files for the maximally different (H1) and optimal reward functions (H2). This notebook also creates all of the parallel coordinate plots for comparison. These are in turn saved in ```../Plots/```. Finally, this file also computes the Kendall's tau rankings for all experiment variations. 
  * ```performance_difference_tests.py``` computes the Hoeffding Bound and Mann Whitney U-test results. 
  
* ```User_Studies```
  * This directory contains code for running the user study, as well as the raw and processed user study data.
  * ```Expert-User-Study/Expert_User_Study.ipynb``` contains the full user study Jupyter notebook. 
  * ```Expert-User-Study/user_tests/*``` contains a directory for each user. Within each directory, there is a ```csv``` file and a ```txt``` file. The ```csv``` file corresponds to every training attempt; the ```txt``` file corresponds to their submitted configuration.
  * ```check_reward_fn_validitiy.py``` uses value iteration to assess whether the user's reward function results in an approximation of the optimal policy
  * ```get_final_reward_fns.py``` extracts all the users' reward functions (either in total or just their submitted versions)
  * ```assess_reward_fn_overfitting.py``` re-trains each user's submitted agent configuration 10 times. The results are saved to ```Experiments/saved_reward_fn_performances/AAAI_Experiments/UserPerformance/overfitting_checks``` 

* ```RL_algorithms```
  * This directory contains all of the code for training the agents. In particular, there are implementations for Q-learning, PPO, A2C, DDQN, and Value Iteration in this directory. 

## Data
Due to file size restrictions, data is available through the anonymous link: https://drive.google.com/file/d/1r4iEVCNsp0F1ox0ylxXGqDd7L31nASu1/view?usp=sharing.
Please download this file and extract its contents to ```Experiments/saved_reward_fn_performances```, leading to the full filepath: ```Experiments/saved_reward_fn_performances/AAAI_Experiments```
* ```Experiments/saved_reward_fn_performances/AAAI_Experiments```
  * ```QLearnBaseline```
    * This contains all 5196 reward function comarisons for the standard hyperparameters, including gamma=0.99 and alpha=0.05
  * ```QLearnAlpha0pt25```
    * This contains all 5196 reward function comarisons for alpha=0.25 experiment
  * ```QLearnGamma0pt8```
    * This contains all 5196 reward function comarisons for gamma=0.8 experiment
  * ```QLearnGamma0pt5```
    * This contains all 5196 reward function comarisons for gamma=0.5 experiment
  * ```TopDiffs```
    * This directory contains CSV files which list the reward functions which result in the maximally different true performance metric scores across experiment variations
    * These CSV files correspond to the Hypothesis 1 analyses. 
  * ```TopOverallFns```
    * This directory contains CSV files which list the reward functions which result in the optimal true performance metric scores for each experimental condition
    * These CSV files correspond to the Hypothesis 2 analyses. 
  * ```TopDiffAndOverallDeepDives```
    * This directory contains all of the data for the reward functions which are listed in the CSV files in the ```TopDiffs``` and ```TopOverallFns``` directories.
  * ```UserPerformance```
    * This directory contains all of the data for the expert users' best-case-valid reward functions' performance with their specified algorithms and hyperparameters.
  * ```UserRewardFnReRunWithDefaultParams```
    * This directory contains all of the data for the expert users' best-case-valid reward functions' performance when RECOMPUTED with our set of standard hyperparameters and algorithms.

Data directly related to expert overfitting is in ```User_Studies/Expert-User-Study/```
* ```user_overfitting.csv``` contains the details of each best-case-valid reward function's performance. 
* ```all_reward_fns.csv``` contains details of all of the reward functions used during the study, across all users.
* ```final_reward_fns.csv``` contains the details of all of the submitted reward functions, across all users.
* ```Expert-User-Study/user_tests/*``` contains a directory for each user. Within each directory, there is a ```csv``` file and a ```txt``` file. The ```csv``` file corresponds to every training attempt; the ```txt``` file corresponds to their submitted configuration.

## Qualitative Analysis: Thematic Analysis 
The annotated transcripts for all user studies are located in ```QualitativeAnalysis/AAAI_2023_Reward_Design_Thematic_Analysis```

default_q_learn_hyper_params = {"alg": "Q_learn",
                                "env_timesteps": 200,
                                "num_environments": 10,
                                "alpha_lr": 0.05,
                                "epsilon": 0.15,
                                "gamma": 0.99,
                                "num_episodes": 2000,
                                "record_freq": 1,
                                "num_tests": 1,
                                }

default_a2c_hyper_params = {"alg": "A2C",
                            "lr": 1e-3,
                            "gamma": 0.99,
                            "max_steps": 200,
                            "num_episodes": 5000,
                            "entropy_coeff": 1e-2,
                            "neural_net_hidden_size": 144,
                            "neural_net_extra_layers": 0,
                            "plot_update_freq": 100,
                            "n_step_update": 20,
                            "num_environments": 10,
                            }

default_ppo_hyper_params = {"alg": "PPO",
                            "num_episodes": 5000,
                            "max_steps": 200,  # max timesteps per episode
                            "update_steps": 200 * 4,  # update policy every n timesteps
                            "K_epochs": 80,  # 80 update policy for K epochs in one PPO update
                            "eps_clip": 0.2,  # clip parameter for PPO
                            "gamma": 0.99,  # discount factor
                            "lr": 5e-3,  # 0.0003 learning rate for actor network
                            "entropy_coeff": 1e-2, # the entropy coefficient,
                            "neural_net_hidden_size": 144,
                            "num_environments": 10,
                            }

default_ddqn_hyper_params = {"alg": "DDQN",
                             "exp_replay_size": 5000,
                             "update_steps": 128,
                             "batch_size": 128,
                             "lr": 1e-3,
                             "sync_frequency": 5,
                             "num_episodes": 5000,
                             "epsilon_decay": 3000,
                             "epsilon_min": 0.15,
                             "test_freq": 100,
                             "plotting_steps": 1000,
                             "gamma": 0.99,
                             "neural_net_hidden_size": 144,
                             "num_environments": 10,
                             }

default_param_lookup = {"Q_Learn": default_q_learn_hyper_params,
                        "A2C": default_a2c_hyper_params,
                        "DDQN": default_ddqn_hyper_params,
                        "PPO": default_ppo_hyper_params,
                        }
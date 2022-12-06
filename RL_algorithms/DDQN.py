import time

import ipywidgets
import numpy as np
import torch
import copy

import random
import gym
from tqdm import tqdm

from gym_hungry_thirsty.envs.hungry_thirsty_reward_fns import *
import gym_hungry_thirsty

from Utils import InteractiveLearningCurvePlot
from Utils import QNetwork
from Utils import ExperienceReplay
from default_parameters import *
import datetime as dt

"""
Original DQN paper: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
Extension to DDQN: https://arxiv.org/pdf/1509.06461.pdf 
Reference: https://github.com/mahakal001/reinforcement-learning/blob/master/cartpole-dqn
Reference: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""


class DDQN_Agent:
    def __init__(self,
                 obs_size,
                 num_actions,
                 device,
                 neural_net_hidden_size=144,
                 lr=1e-3,
                 sync_freq=5,
                 exp_replay_size=256,
                 gamma=0.95):
        """

        :param obs_size: int
        :param num_actions: int
        :param device: torch device type (typically torch.device('cpu') or torch.device('cuda:0'))
        :param neural_net_hidden_size: int
        :param lr: float
        :param sync_freq: int
        :param exp_replay_size: int
        :param gamma: float
        """
        self.num_actions = num_actions
        self.device = device
        self.q_net = QNetwork(obs_size=obs_size,
                              num_actions=num_actions,
                              hidden_size=neural_net_hidden_size).to(self.device)
        self.q_target_net = copy.deepcopy(self.q_net).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        for param in self.q_target_net.parameters():
            param.requires_grad = False

        self.network_sync_freq = sync_freq
        self.network_sync_counter = 0
        self.gamma = torch.tensor(gamma).float()
        self.experience_replay = ExperienceReplay(exp_replay_size=exp_replay_size)
        self.best_mean_performance = None
        return

    def load_pretrained_model(self, model_path):
        """
        Load a pretrained model

        :param model_path: string
        :return: None, load into memory
        """
        self.q_net.load_state_dict(torch.load(model_path))

    def save_trained_model(self, model_path="ddqn-model.pth"):
        """
        Save the network weights

        :param model_path: string
        :return:
        """
        torch.save(self.q_net.state_dict(), model_path)

    def get_action(self, state, epsilon):
        """
        Given a state, apply an e-greedy strategy to select an action

        :param state: np array
        :param epsilon: float
        :return:
        """
        with torch.no_grad():
            Qp = self.q_net(torch.from_numpy(state).float().to(self.device))
            if random.random() > epsilon:
                _, action = torch.max(Qp, dim=0)
            else:
                action = torch.randint(0, self.num_actions, (1,))
        return action

    def train(self, batch_size, loss_fn=torch.nn.MSELoss()):
        """

        :param batch_size:
        :param loss_fn: the torch loss function; defaults to MSE
        :return:
        """
        # update target network every self.network_sync_freq steps
        if self.network_sync_counter == self.network_sync_freq:
            self.q_target_net.load_state_dict(self.q_net.state_dict())
            self.network_sync_counter = 0
        self.network_sync_counter += 1

        # sample from replay buffer
        states, actions, rewards, next_states, dones = \
            self.experience_replay.sample_from_experience(sample_size=batch_size)
        actions = actions.to(torch.int64).to(self.device)
        states = states.to(self.device)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), dtype=torch.bool)
        non_final_next_states = [s for s in next_states if s is not None]
        non_final_next_states = torch.stack(non_final_next_states).to(self.device)

        # predict expected return of current state using current network - Q(s, a | theta)
        # states - a tensor of batchsize
        # actions - also a tensor of batchsize. unsqueeze converts [1, 1, 0, 3, ...] to [[1],[1],[0],[3],...]
        # gather - get the q_net's prediction for each state,action pair
        # compute Q(s, a)
        state_action_values = self.q_net(states).gather(dim=1, index=actions.unsqueeze(1))

        # get target return using target network
        # initialize all next_state_values to 0; only update non-terminal next_state_values to be non-zero
        # compute max_a Q_target(s+1, a)
        next_state_values = torch.zeros(batch_size, device=self.device)
        next_state_values[non_final_mask] = self.q_target_net(non_final_next_states).max(dim=1)[0].detach()

        # compute loss & step
        expected_state_action_values = rewards.to(self.device) + self.gamma * next_state_values
        loss = loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test_performance(self, env, agent, epsilon, num_tests=5, metric='reward', reward_fn=None):
        """

        :param env:
        :param agent:
        :param num_tests:
        :param metric: string, either 'reward' or 'fitness' TODO add return
        :param reward_fn: optional, the user defined reward function
        :return:
        """
        performances = []
        fitnesses = []

        for _ in range(num_tests):
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_fitness = 0
            while not done:
                state_as_arr = env.get_state_as_np_array() if "hungry-thirsty-v0" in env.spec.id else obs
                A = agent.get_action(state=state_as_arr, epsilon=epsilon)
                obs_next, reward, done, _ = env.step(A.item())

                if reward_fn is not None:
                    reward = reward_fn(state=obs, action=A.item(), new_state=obs_next)
                    if not obs["hungry"]:
                        episode_fitness += 1

                episode_reward += reward
                obs = obs_next

                if done:
                    performances.append(episode_reward)
                    fitnesses.append(episode_fitness)

        if metric == 'reward':
            if self.best_mean_performance is None or np.mean(performances) > self.best_mean_performance:
                self.best_mean_performance = np.mean(performances)
                self.save_trained_model(model_path='models/ddqn-best-{}.pth'.format(env.spec.id))
        elif metric == 'fitness':
            if self.best_mean_performance is None or np.mean(fitnesses) > self.best_mean_performance:
                self.best_mean_performance = np.mean(fitnesses)
                self.save_trained_model(model_path='models/ddqn-best-{}.pth'.format(env.spec.id))

        return performances, fitnesses


def create_ddqn_agent(env,
                      hyper_params,
                      user_reward_fn=None,
                      plot_state_visit_distribution=False,
                      plot_results=True,
                      device=torch.device('cpu')):
    """

    :param env:
    :param hyper_params:
    :param user_reward_fn:
    :param plot_state_visit_distribution: bool, whether to plot a heatmap of the state visits
    :param plot_results: bool, whether to plot the training curve
    :param device: torch device (cpu or cuda)
    :return:
    """
    assert ("lr" in hyper_params.keys())
    assert ("update_steps" in hyper_params.keys())
    assert ("batch_size" in hyper_params.keys())
    assert ("epsilon_min" in hyper_params.keys())
    assert ("epsilon_decay" in hyper_params.keys())
    assert ("num_episodes" in hyper_params.keys())
    assert ("gamma" in hyper_params.keys())
    assert ("exp_replay_size" in hyper_params.keys())
    assert ("sync_frequency" in hyper_params.keys())
    assert ("neural_net_hidden_size" in hyper_params.keys())
    learning_rate = hyper_params["lr"]
    update_steps = hyper_params["update_steps"]
    batch_size = hyper_params["batch_size"]
    epsilon_min = hyper_params["epsilon_min"]
    epsilon_decay = hyper_params["epsilon_decay"]
    num_episodes = hyper_params["num_episodes"]
    gamma = hyper_params["gamma"]
    exp_replay_size = hyper_params["exp_replay_size"]
    sync_frequency = hyper_params["sync_frequency"]
    neural_net_hidden_size = hyper_params["neural_net_hidden_size"]

    epsilon = 1 # epsilon starts at 1, and decays as epsilon -= (1 / epsilon_decay) until it reaches epsilon_min
    plotting_steps = 100
    plot_interactively = True

    # set up tracking and visuals for state visit history
    state_history = {}
    state_dist_viewer = None

    agent = DDQN_Agent(obs_size=env.observation_space.shape[0],
                       num_actions=env.action_space.n,
                       neural_net_hidden_size=neural_net_hidden_size,
                       lr=learning_rate,
                       sync_freq=sync_frequency,
                       exp_replay_size=exp_replay_size,
                       device=device,
                       gamma=gamma)

    # Tracking
    reward_list = []
    fitness_list = []

    total_steps = 0
    num_updates = 0

    if plot_results:
        plotting = {
            0: ("Not Hungry Count Per Episode\n" +
                r"$\Sigma_{(s, a, s') \backsim \tau} \mathbb{1}(\mathrm{s[hungry] = False)}$",
                "fitness_list", ("Episode", "Not Hungry Count"), True),
            1: ("Undiscounted Return\n" +
                r"Summed Reward Per Episode: $\Sigma_{(s, a, s') \backsim \tau} r'(s)$",
                "reward_list", ("Episode", "Return"), False),
        }
        fig = InteractiveLearningCurvePlot(num_axes=len(plotting.items()))

    if state_dist_viewer is None:
        state_dist_viewer = ipywidgets.Image()
        from IPython import display
        display.display(state_dist_viewer)

    try:
        already_plotted_dist = False
        for episode_idx in tqdm(range(num_episodes)):
            obs = env.reset()
            done = False
            losses = 0
            ep_len = 0
            rew = 0
            fitness = 0
            show_episode = False
            while not done:
                ep_len += 1
                if "hungry-thirsty-v0" in env.spec.id:
                    state_as_arr = env.get_state_as_np_array()
                    if not obs["hungry"]:
                        fitness += 1

                    # plot state distribution
                    if plot_state_visit_distribution:
                        if (episode_idx % 100 == 0 and episode_idx > 0) or \
                                (episode_idx == 0 and not already_plotted_dist):
                            env.render_state_distribution(state_dist_viewer, state_visit_dict=state_history)
                            already_plotted_dist = True

                    if obs['position'] in state_history.keys():
                        state_history[obs['position']] += 1
                    else:
                        state_history[obs['position']] = 1
                else:
                    state_as_arr = obs

                A = agent.get_action(state=state_as_arr, epsilon=epsilon)
                obs_next, reward, done, _ = env.step(A.item())

                if user_reward_fn is not None:
                    reward = user_reward_fn(state=obs, action=A.item(), new_state=obs_next)

                if "hungry-thirsty-v0" in env.spec.id:
                    next_state_as_arr = env.get_state_as_np_array()
                else:
                    next_state_as_arr = obs_next

                agent.experience_replay.collect_experience(state=state_as_arr,
                                                           action=A.item(),
                                                           reward=reward,
                                                           next_state=next_state_as_arr,
                                                           done=done)

                obs = obs_next
                rew += reward
                total_steps += 1

                if total_steps % update_steps == 0 and len(agent.experience_replay) > batch_size:
                    for _ in range(4):  # for 75000 steps, 4 * 60 seems to work better
                        num_updates += 1
                        loss = agent.train(batch_size=batch_size)
                        losses += loss

            reward_list.append([episode_idx, [rew]])
            fitness_list.append([episode_idx, [fitness]])

            # decay epsilon (decrease exploration tendency)
            if epsilon > epsilon_min:
                epsilon -= (1 / epsilon_decay)

            # update the plot
            if plot_results:
                if plot_interactively and episode_idx % plotting_steps == 0 or episode_idx == num_episodes - 1:
                    for idx in plotting.keys():
                        title, list_name, labels, draw_scaling_lines = plotting[idx]
                        plotting_data = locals()[list_name]
                        if len(plotting_data) > 1:
                            fig.update_subplot(axis_id=idx,
                                               title=title,
                                               learning_performance=plotting_data,
                                               labels=labels,
                                               draw_scaling_lines=draw_scaling_lines,
                                               )
    except:
        return agent, epsilon, reward_list, fitness_list

    if plot_results:
        fig.end_interactive()
    return agent, epsilon, reward_list, fitness_list


def run_episode(env, agent, epsilon, render=True, jupyter=False, canvas=None):
    """

    :param env:
    :param agent:
    :param epsilon: float, the e-greedy selection probability
    :param render: bool, whether to render the scene
    :param jupyter: bool, whether called from Jupyter
    :param canvas: if called from jupyter, this is an ipywidget canvas
    :return:
    """
    # agent.load_pretrained_model(model_path='models/ddqn-best-{}.pth'.format(env.spec.id))
    with torch.no_grad():
        obs = env.reset()
        done = False
        fitness = 0
        while not done:
            if "hungry-thirsty" in env.spec.id:
                state_as_arr = env.get_state_as_np_array()
                if not obs["hungry"]:
                    fitness += 1
                if render and not jupyter:
                    env.render(score=fitness)
                    time.sleep(0.1)
                elif render and jupyter:
                    env.jupyter_render(canvas, score=fitness)
                    time.sleep(0.1)
            else:
                state_as_arr = obs
                if render:
                    env.render()
                    time.sleep(0.1)

            A = agent.get_action(state=state_as_arr, epsilon=epsilon)
            obs_next, reward, done, _ = env.step(A.item())
            obs = obs_next
    env.close()


def main():
    # for some reason, cpu is faster; leaving this line in here to switch
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    # user input to select environment
    environments = {0: 'CartPole-v1',
                    1: 'hungry-thirsty-v0'}
    print("Choose an environment (0, 1, ...): " + str(environments))
    env_name = environments[int(input())]

    for _ in range(5):

        time_1 = dt.datetime.now()
        env = gym.make(env_name, size=(4,4))
        env.reset()
        env.reset(food_loc=(0,0), water_loc=(0,3))

        if "hungry-thirsty" in env.spec.id:
            env_timestep_limit = 200
            env.update_step_limit(env_timestep_limit)

        hyper_params = default_ddqn_hyper_params

        agent, epsilon, _, _ = create_ddqn_agent(env=env,
                                                 hyper_params=hyper_params,
                                                 user_reward_fn=wdrcf_reward_fn,
                                                 device=device)

        time_diff = dt.datetime.now() - time_1
        print ("TIME DIFF: ", time_diff.total_seconds())

    while True:
        print("See the agent? Y/N")
        if input() == "y":
            run_episode(env=env, agent=agent, epsilon=hyper_params["epsilon_min"], render=True)
        else:
            break


if __name__ == "__main__":
    main()

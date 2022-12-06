import time
import random

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from collections import deque
import scipy.stats
from os import listdir

""" 
***************************************
Visualization code
    plot_learning_curve(...)
    moving_average(...)
    tsplot(...)
    InteractiveLearningCurvePlot(...) 
***************************************
"""


def plot_learning_curve(learning_performance, trend_line=True):
    """
    Plot the learning curve

    :param learning_performance: an array of arrays [[0, [s_1,...,s_n]],
                                                     ...,
                                                     [M, [s_1,...,s_n]]
    :param trend_line: boolean, whether to plot trend line
    :return: None
    """
    assert (len(learning_performance) > 1)
    assert (len(learning_performance[0]) == 2)

    # extract x_axis labels (e.g., training episode IDs) and scores
    x_axis = [x[0] for x in learning_performance]
    scores = [x[1] for x in learning_performance]

    # compute the mean and stddev, and plot both
    mean_perf = np.mean(scores, axis=1)
    std_dev_perf = np.std(scores, axis=1)
    plt.plot(x_axis, mean_perf)
    plt.fill_between(x_axis, mean_perf - std_dev_perf,
                     mean_perf + std_dev_perf, alpha=0.3)

    if trend_line:
        z = np.polyfit(x_axis, mean_perf, 1)
        p = np.poly1d(z)
        plt.plot(x_axis, p(x_axis), "r--", label='trend line')
        plt.legend()

    plt.show()


def moving_average(a, n=10):
    """
    Compute a moving average over the data

    :param a: the data
    :param n: int, the window size
    :return: a numpy array
    """
    trend_line = np.convolve(a, np.ones(n) / n, mode='same')
    if len(trend_line) > 50:
        trend_line[0:25] = trend_line[25]
        trend_line[-25:] = trend_line[-25]
    return trend_line


def tsplot(ax, data, plot_ci=True, **kw):
    """
    Plot a line with CI (+/- stddev)

    :param ax: matplotlib axis
    :param data: data to plot; a list of lists
    :param plot_ci: optional, whether to plot the confidence interval
    :param kw: any plotting args
    :return: None
    """
    x = np.arange(data.shape[0])
    est = np.mean(data, axis=-1)
    if plot_ci:
        sd = np.std(data, axis=-1)
        cis = (est - sd, est + sd)
        ax.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)
    ax.plot(x, est, **kw)
    ax.margins(x=0)


class InteractiveLearningCurvePlot:
    """
    Interactively plot data
    """

    def __init__(self, num_axes):
        """
        initialize the plot

        :param num_axes: int. the number of subplots to include
        """
        plt.ion()
        self.fig, self.axes = plt.subplots(num_axes, figsize=(12, 12))
        self.lines = []
        self.trend_lines = []

        self.y_cur = []
        for axis in self.axes:
            self.y_cur.append(10)
            line, = axis.plot([0], [0], label='raw data')
            trend_line, = axis.plot([0], [0], label='weighted average')
            self.lines.append(line)
            self.trend_lines.append(trend_line)
            axis.set_title("undefined")
            axis.legend()

    def update_subplot(self,
                       axis_id,
                       title,
                       learning_performance,
                       labels=None,
                       draw_scaling_lines=False,
                       draw_trend_lines=True):
        """
        update the plot and re-draw

        :param axis_id: int, the axis lookup id
        :param title: string, the plot title
        :param learning_performance: an array of arrays [[0, [s_1,...,s_n]],
                                                        ...,
                                                        [M, [s_1,...,s_n]]
        :param labels: the axis labels
        :param draw_scaling_lines: bool, whether to draw the scaling lines,
        :param draw_trend_lines: bool, whether to apply a weighted average to the data
        :return: None
        """
        x_axis_label, y_axis_label = "", ""
        if labels is not None:
            x_axis_label, y_axis_label = labels

        x_axis = [x[0] for x in learning_performance]
        scores = [np.mean(x[1]) for x in learning_performance]

        self.axes[axis_id].set_title(title, fontsize=18)
        self.axes[axis_id].set_xlabel(x_axis_label, fontsize=14)
        self.axes[axis_id].set_ylabel(y_axis_label, fontsize=14)
        self.axes[axis_id].tick_params(axis='x', labelsize=14)
        self.axes[axis_id].tick_params(axis='y', labelsize=14)

        line = self.lines[axis_id]
        line.set_xdata(x_axis)
        line.set_ydata(scores)

        # dynamically update axis limits w.r.t. plotting data
        self.axes[axis_id].relim()
        self.axes[axis_id].autoscale_view()

        if draw_scaling_lines:
            _, ymax = self.axes[axis_id].get_ylim()
            while self.y_cur[axis_id] <= ymax:
                self.axes[axis_id].axhline(y=self.y_cur[axis_id], color='orange', linestyle='--', alpha=0.3)
                self.y_cur[axis_id] += 10

        if draw_trend_lines:
            trend_line = self.trend_lines[axis_id]
            trend_line.set_xdata(x_axis)
            trend_line.set_ydata(moving_average(scores, n=30))

        self.fig.tight_layout()
        self.fig.canvas.draw()
        time.sleep(0.1)
        self.fig.canvas.flush_events()

        # plt.show()

    @staticmethod
    def end_interactive():
        """
        Turn off interactive mode

        :return:
        """
        plt.ioff()
        plt.show()


""" 
***************************************
Misc
    cum_discounted_rewards(...)
    mean_confidence_interval(...)
    split_data(...)
    find_filenames_with_extension(...) 
***************************************
"""


def cum_discounted_rewards(rewards, gamma, whitening=False):
    """
    Compute the cumulative discounted rewards for each timestep of the trajectory

    :param rewards: an array, the length of an episode, consisting of rewards
    :param gamma: float, corresponding to discount factor
    :param whitening: bool, corresponding to whether to perform whitening or not
    :return: G, an array of cumulative discounted rewards over time
    """
    G = []
    total_reward = 0
    for r in reversed(rewards):
        total_reward = r + total_reward * gamma
        G.insert(0, total_reward)
    G = torch.tensor(G)

    if whitening:
        G = (G - G.mean()) / G.std()

    return G


def mean_confidence_interval(data, confidence=0.95):
    """
    Compute the confidence interval for some data

    :param data: an array of data
    :param confidence: float, the confidence bound
    :return:
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a, axis=-1), scipy.stats.sem(a, axis=-1)
    h = se * scipy.stats.t.ppf((1 - confidence) / 2., n-1)
    return m, m-h, m+h


def split_data(data, advesarial=True):
    """
    Split some data in the form [datum_1, datum_2, datum_3, ...] into two lists
    e.g., [datum_1, datum_3], [datum_2, ...]

    :param data: a list of arbitrary data
    :param advesarial: boolean. If true, split data by the last value of each datum
    :return: list_1, list_2 (two splits of the data)
    """
    abbreviated_data = [(idx, datum[-1]) for idx, datum in enumerate(data)]
    if advesarial:
        abbreviated_data.sort(key=lambda y: y[1])
    else:
        random.shuffle(abbreviated_data)
    indeces = [idx for idx, datum in abbreviated_data]
    half_pt = int(len(indeces)/2)
    return [data[idx] for idx in indeces[:half_pt]], \
           [data[idx] for idx in indeces[half_pt:]]


def find_filenames_with_extension(path_to_dir, extension):
    """
    Return all the CSV files in a folder
    :param path_to_dir: string, the path to a directory
    :param extension: string, e.g. ".csv"
    :return: list
    """
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(extension)]


""" 
***************************************
Experience Replay
RolloutBuffer
***************************************
"""


class ExperienceReplay:
    def __init__(self, exp_replay_size):
        """
        Initialize an experience replay buffer of size exp_replay_size

        :param exp_replay_size: int, the size of the buffer
        """
        self.experience = deque(maxlen=exp_replay_size)

    def collect_experience(self, state, action, reward, next_state, done):
        """
        Add experience to the buffer

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return: None
        """
        self.experience.append([state, action, reward, next_state, done])

    def sample_from_experience(self, sample_size):
        """
        Sample experience from replay buffer

        :param sample_size: int, the max size
        :return: states, actions, rewards, next states, dones
        """
        if len(self.experience) < sample_size:
            sample_size = len(self.experience)
        sample = random.sample(self.experience, sample_size)
        s = torch.tensor(np.array([exp[0] for exp in sample])).float()
        a = torch.tensor([exp[1] for exp in sample]).float()
        rn = torch.tensor([exp[2] for exp in sample]).float()
        sn = torch.tensor(np.array([exp[3] for exp in sample])).float()
        dones = torch.tensor([exp[4] for exp in sample]).float()
        return s, a, rn, sn, dones

    def __len__(self):
        """
        Compute the length of the buffer
        :return:
        """
        return len(self.experience)


class RolloutBuffer:
    def __init__(self):
        """
        Initialize memory buffers - one for each object we want to track
        """
        self.states = []
        self.actions = []
        self.logprobs = []
        self.entropy = []
        self.rewards = []
        self.is_terminals = []
        self.value_estimates = []

    def add_experience(self, action, state, logprob, entropy, reward, done, value_estimate=None):
        """
        Add experience to the buffers

        :param action: tensor
        :param state: tensor
        :param logprob: tensor
        :param entropy: tensor
        :param reward: float
        :param done: bool
        :param value_estimate: float

        :return: None
        """
        self.states.append(torch.FloatTensor(state))
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.entropy.append(entropy)
        self.rewards.append(reward)
        self.is_terminals.append(done)
        self.value_estimates.append(value_estimate)

    def clear(self):
        """
        Delete contents of memory buffers. This is done after updates.

        :return:
        """
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.entropy[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.value_estimates[:]


    def _zip(self):
        return zip(self.states,
                   self.actions,
                   self.logprobs,
                   self.entropy,
                   self.rewards,
                   self.is_terminals,
                   self.value_estimates)

    def reversed(self):
        for data in list(self._zip())[::-1]:
            yield data

    def __len__(self):
        return len(self.actions)


""" 
***************************************
Network Initializations
    QNetwork
    ValueNetwork
    ActorNetwork 
    PolicyNetwork (same as ActorNetwork)
    CriticNetwork (same as ValueNetwork)
***************************************
"""


class QNetwork(nn.Module):
    """
    The network architecture
    """

    def __init__(self, obs_size, num_actions, hidden_size=64, device=torch.device("cpu")):
        """

        :param obs_size: int
        :param num_actions: int
        :param hidden_size: int
        :param device: torch device
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_actions)
        self.device = device

    def forward(self, x):
        """

        :param x: numpy array or tensor of size obs_size
        :return:
        """
        x = self.fc1(x.to(self.device))
        x = self.tanh(x)
        return self.fc2(x)


class ValueNetwork(nn.Module):
    """
    The value / critic network
    """

    def __init__(self, obs_size, hidden_size=64, extra_layers=0, device=torch.device("cpu")):
        """
        :param obs_size: int; the size of the state / input
        :param hidden_size: int; the size of the hidden layer
        :param extra_layers: int, the number of additional layers
        :param device: the torch device (cpu or cuda)
        """
        super(ValueNetwork, self).__init__()
        # value network
        self.value_fc1 = torch.nn.Linear(obs_size, hidden_size)
        self.value_fc2 = torch.nn.Linear(hidden_size, 1)
        self.extra_layers = []
        for _ in range(extra_layers):
            self.extra_layers.append(torch.nn.Linear(hidden_size, hidden_size))
        self.relu = torch.nn.ReLU()
        self.device = device

    def forward(self, state):
        """
        Compute the state value (value network)

        :param state: a numpy array
        :return:
        """
        x = self.value_fc1(state.to(self.device))
        x = self.relu(x)
        for layer in self.extra_layers:
            x = layer(x)
            x = self.relu(x)
        state_value = self.value_fc2(x)
        return state_value

    def critique(self, state, action, actor):
        """
        Run both actor & critic networks

        :param state: state tensor
        :param action: action tensor
        :param actor: ActorNetwork
        :return: action_log_probs: an array
                 state_values: a float
                 dist_entropy: an array
        """
        action_probs = actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.forward(state)
        return action_logprobs, state_values, dist_entropy


class PolicyNetwork(nn.Module):
    """
    The policy / actor network
    """

    def __init__(self, obs_size, num_actions, hidden_size=64, extra_layers=0, device=torch.device("cpu")):
        """
        :param obs_size: int; the size of the state / input
        :param num_actions: int; the size of the output / number of actions
        :param hidden_size: int; the size of the hidden layer
        :param extra_layers: int, the number of additional layers
        :param device: torch device
        """
        super(PolicyNetwork, self).__init__()
        # policy / action network
        self.first_layer = torch.nn.Linear(obs_size, hidden_size)
        self.extra_layers = []
        for _ in range(extra_layers):
            self.extra_layers.append(torch.nn.Linear(hidden_size, hidden_size))
        self.last_layer = torch.nn.Linear(hidden_size, num_actions)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.device = device

    def forward(self, state):
        """
        Compute the action probs (action network)

        :param state: a numpy array
        :return:
        """
        # action network
        x = self.first_layer(state.to(self.device))
        x = self.relu(x)

        for layer in self.extra_layers:
            x = layer(x)
            x = self.relu(x)

        x = self.last_layer(x)
        action_probs = self.softmax(x)
        return action_probs

    def get_action(self, state, return_entropy=False, deterministic=False):
        """
        Compute the action to take, corresponding log probability, and the state value

        :param state: a numpy array
        :param return_entropy: boolean, whether to return entropy of dist
        :param deterministic: boolean. If true, return max action_probs (rather than sampling from a distribution)
        :return: action, action_log_prob, action_probs
        """
        action_probs = self.forward(state)
        dist = torch.distributions.Categorical(action_probs)
        if deterministic:
            action = torch.argmax(action_probs)
        else:
            action = dist.sample()
        action_log_prob = dist.log_prob(action)
        if return_entropy:
            entropy = dist.entropy().mean()
            return action.detach(), action_log_prob, entropy
        else:
            return action.detach(), action_log_prob


# Provide aliases for networks
ActorNetwork = PolicyNetwork
CriticNetwork = ValueNetwork


class ActorCriticNetwork(nn.Module):
    """
    Jointly trained actor critic network
    """
    def __init__(self, obs_size, num_actions, device=torch.device('cpu'), hidden_size=64, extra_layers=0):
        """
        Initialize the actor critic network

        :param obs_size:
        :param num_actions:
        :param device:
        :param hidden_size:
        :param extra_layers:
        """
        super(ActorCriticNetwork, self).__init__()
        self.actor = ActorNetwork(obs_size=obs_size,
                                  num_actions=num_actions,
                                  hidden_size=hidden_size,
                                  device=device,
                                  extra_layers=extra_layers)
        self.critic = CriticNetwork(obs_size=obs_size,
                                    hidden_size=hidden_size,
                                    device=device,
                                    extra_layers=extra_layers)
        self.device = device

    def forward(self, state):
        """
        Run the actor critic network

        :param state:
        :return:
        """
        action, action_log_prob = self.actor.get_action(state.to(self.device))
        state_value = self.critic(state.to(self.device))
        return state_value, action, action_log_prob

    def copy_state_dict(self, other_actor_critic_network):
        """
        Copy another network's parameters into this network

        :param other_actor_critic_network: an ActorCriticNetwork of the same class
        :return:
        """
        self.actor.load_state_dict(other_actor_critic_network.actor.state_dict())
        self.critic.load_state_dict(other_actor_critic_network.critic.state_dict())

import time

import gym
import ipywidgets
import torch
import torch.nn as nn
from tqdm import tqdm

from Utils import InteractiveLearningCurvePlot
from Utils import ActorCriticNetwork
from Utils import RolloutBuffer
from gym_hungry_thirsty.envs.hungry_thirsty_reward_fns import *
import datetime as dt
from default_parameters import *
import gym_hungry_thirsty

"""
Paper: https://arxiv.org/abs/1707.06347
Code reference: https://github.com/nikhilbarhate99/PPO-PyTorch
"""


class PPO:
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 lr=0.001,
                 gamma=0.99,
                 k_epochs=80,
                 eps_clip=0.2,
                 network_hidden_layer_size=144):
        """
        Initialize the PPO agent

        :param state_dim: int, the size of the input
        :param action_dim: int, the size of the (discrete) action space
        :param device: torch device type
        :param lr: float, the learning rate
        :param gamma: float, the discount factor
        :param k_epochs: int, the number of epochs between updates
        :param eps_clip: float, the trust region clipping factor
        :param network_hidden_layer_size: int, the size of the hidden layer(s) in the network
        """
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.rollout_buffer = RolloutBuffer()

        # main network
        self.actor_critic = ActorCriticNetwork(obs_size=state_dim,
                                               num_actions=action_dim,
                                               device=device,
                                               hidden_size=network_hidden_layer_size)

        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)
        # and target network
        self.target_actor_critic = ActorCriticNetwork(obs_size=state_dim,
                                                      num_actions=action_dim,
                                                      device=device,
                                                      hidden_size=network_hidden_layer_size)
        self.target_actor_critic.copy_state_dict(other_actor_critic_network=self.actor_critic)

        self.loss = nn.MSELoss()

        self.device = device
        self.actor_critic.actor.to(device)
        self.actor_critic.critic.to(device)
        self.target_actor_critic.actor.to(device)
        self.target_actor_critic.critic.to(device)

    def select_action(self, state):
        """
        Run the ppo agent target actor to select the next action

        :param state: an np array
        :return:
        """
        with torch.no_grad():
            state = torch.FloatTensor(state)
            action, action_log_probs = self.target_actor_critic.actor.get_action(state.to(self.device))

        return action, action_log_probs

    def update(self, c1=0.5, entropy_coeff=0.01):
        """
        Update the actor and critic networks

        :param c1: float, hyperparameter controlling the importance of the critic's state value estimates
        :param entropy_coeff: float, hyperparameter controlling the importance of the entropy penalty
        :return:
        """
        # get discounted cum returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.rollout_buffer.rewards),
                                       reversed(self.rollout_buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # normalize the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        rewards = rewards.to(self.device)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.rollout_buffer.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(self.rollout_buffer.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.rollout_buffer.logprobs, dim=0)).detach()

        for _ in range(self.k_epochs):
            logprobs, state_values, dist_entropy = self.actor_critic.critic.critique(state=old_states,
                                                                                     action=old_actions,
                                                                                     actor=self.actor_critic.actor)
            state_values = torch.squeeze(state_values)
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + c1 * self.loss(state_values, rewards) - entropy_coeff * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.target_actor_critic.copy_state_dict(other_actor_critic_network=self.actor_critic)
        self.rollout_buffer.clear()


def train_ppo_agent(env,
                    hyper_params,
                    device=torch.device("cpu"),
                    user_reward_fn=None,
                    plot_results=True,
                    plot_state_visit_distribution=None,
                    ):
    """
    :param env: the environment
    :param hyper_params: a dict which must contain:
                            max_ep_len: int, the maximum episode length
                            num_epsiodes: int, the number of episodes
                            update_steps: int, the update frequency
                            K_epochs: int, the number of epochs to run before updating the target network
                            eps_clip: float, the clipping factor for trust region updates
                            gamma: float, the discount factor
                            lr_actor: float, the learning rate for the actor network
                            lr_critic: float, the learning rate for the critic network
    :param device: the torch device
    :param user_reward_fn: function, which computes the reward
    :param plot_results: bool, whether to plot the learning curve in real time or not
    :param plot_state_visit_distribution: bool, whether to plot state visit dist
    :return: ppo_agent, a PPO class
    """
    assert ("max_steps" in hyper_params.keys())
    assert ("lr" in hyper_params.keys())
    assert ("gamma" in hyper_params.keys())
    assert ("update_steps" in hyper_params.keys())
    assert ("num_episodes" in hyper_params.keys())
    assert ("eps_clip" in hyper_params.keys())
    assert ("K_epochs" in hyper_params.keys())
    assert ("entropy_coeff" in hyper_params.keys())
    assert ("neural_net_hidden_size" in hyper_params.keys())

    max_steps = hyper_params["max_steps"]
    lr = hyper_params["lr"]
    gamma = hyper_params["gamma"]
    update_steps = hyper_params["update_steps"]
    num_episodes = hyper_params["num_episodes"]
    eps_clip = hyper_params["eps_clip"]
    K_epochs = hyper_params["K_epochs"]
    entropy_coeff = hyper_params["entropy_coeff"]
    neural_net_hidden_size = hyper_params["neural_net_hidden_size"]

    reward_list = []
    fitness_list = []

    total_steps = 0
    num_updates = 0

    if plot_results:
        plotting = {
            0: ("Not Hungry Count Per Episode\n" +
                r"$\Sigma_{(s, a, s') \in \tau} \mathbb{1}(s\mathrm{[is\_hungry]=False)}$",
                "fitness_list", ("Episode", "Not Hungry Count"), True),
            1: ("Undiscounted Return\n" +
                r"Summed Reward Per Episode: $\Sigma_{(s, a, s') \in \tau} r'(s)$",
                "reward_list", ("Episode", "Return"), False),
        }

        fig = InteractiveLearningCurvePlot(num_axes=len(plotting.items()))

    state_dist_viewer = None

    # state space dimension
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    ppo_agent = PPO(state_dim=state_dim,
                    action_dim=action_dim,
                    lr=lr,
                    gamma=gamma,
                    k_epochs=K_epochs,
                    eps_clip=eps_clip,
                    device=device,
                    network_hidden_layer_size=neural_net_hidden_size)

    time_step = 0
    i_episode = 0

    num_updates = 0

    state_history = {}
    if state_dist_viewer is None:
        state_dist_viewer = ipywidgets.Image()
        from IPython import display
        display.display(state_dist_viewer)
    # training loop
    episode_idx = 0
    already_plotted_dist = False
    try:
        for episode_idx in tqdm(range(num_episodes)):
            state = env.reset()
            current_ep_reward = 0
            ep_fitness = 0

            done = False
            while not done:
                # select action with policy
                if "hungry-thirsty" in env.spec.id:
                    # get state as np array
                    state_as_arr = env.get_state_as_np_array()

                    # check fitness
                    if not state["hungry"]:
                        ep_fitness += 1

                    # plot state distribution
                    if plot_state_visit_distribution:
                        if (episode_idx % 100 == 0 and episode_idx > 0) or \
                                (episode_idx == 0 and not already_plotted_dist):
                            env.render_state_distribution(state_dist_viewer, state_visit_dict=state_history)
                            already_plotted_dist = True

                    # track states
                    if state['position'] in state_history.keys():
                        state_history[state['position']] += 1
                    else:
                        state_history[state['position']] = 1

                else:
                    state_as_arr = state
                action, action_log_probs = ppo_agent.select_action(state=state_as_arr)
                new_state, reward, done, _ = env.step(action.item())

                if user_reward_fn is not None:
                    reward = user_reward_fn(state=state, action=action.item(), new_state=new_state)

                # saving reward and is_terminals
                ppo_agent.rollout_buffer.add_experience(action=action,
                                                        state=torch.FloatTensor(state_as_arr),
                                                        logprob=action_log_probs,
                                                        reward=reward,
                                                        done=done,
                                                        entropy=None)

                state = new_state
                time_step += 1
                current_ep_reward += reward

                # update PPO agent
                if time_step % update_steps == 0:
                    num_updates += 1
                    ppo_agent.update(entropy_coeff=entropy_coeff)

                # update the plot
                if plot_results and time_step != 0 and time_step % 10000 == 0:
                    for idx in plotting.keys():
                        title, list_name, labels, draw_line = plotting[idx]
                        plotting_data = locals()[list_name]
                        if len(plotting_data) > 1:
                            fig.update_subplot(axis_id=idx,
                                               title=title,
                                               learning_performance=plotting_data,
                                               labels=labels,
                                               draw_scaling_lines=draw_line)

                if done:
                    reward_list.append([episode_idx, [current_ep_reward]])
                    fitness_list.append([episode_idx, [ep_fitness]])
                    break
    except:
        return ppo_agent, reward_list, fitness_list

    if plot_results:
        fig.end_interactive()
    return ppo_agent, reward_list, fitness_list


def run_episode(env, ppo_agent, render=True, jupyter=False, canvas=None, reward_fn=None):
    """
    Run and visualize an episode of play

    :param env: the environment
    :param ppo_agent: a PPO object
    :param render: boolean, whether to render
    :param jupyter: boolean, whether we're using jupyter notebook or not
    :param canvas: ipythonwidget canvas, only used when jupyter is true
    :param reward_fn: the reward function, if specified
    :return: float, the total undiscounted reward
    """
    if jupyter:
        assert (canvas is not None)

    state = env.reset()
    done = False
    ep_reward = 0
    ep_fitness = 0

    while not done:
        if "hungry-thirsty-v0" in env.spec.id:
            state_as_arr = env.get_state_as_np_array()
            if not state["hungry"]:
                ep_fitness += 1
        else:
            state_as_arr = state

        if render and not jupyter:
            env.render(score=ep_fitness)
            time.sleep(0.1)
        elif render and jupyter:
            env.jupyter_render(canvas, score=ep_fitness)
            time.sleep(0.1)

        # select action with policy
        action, _ = ppo_agent.select_action(state=state_as_arr)
        new_state, reward, done, _ = env.step(action.item())
        if reward_fn is not None:
            reward = reward_fn(state=state_as_arr, action=action.item(), new_state=new_state)
        if reward is not None:
            ep_reward += reward
        state = new_state

    env.close()
    return ep_reward


def create_ppo_agent(env, hyper_params, user_reward_fn, plot_state_visit_distribution=False, plot_results=True):
    """

    :param env: openai gym environment
    :param hyper_params: a dict which must contain:
                            max_ep_len: int, the maximum episode length
                            num_epsiodes: int, the number of episodes
                            update_steps: int, the update frequency
                            K_epochs: int, the number of epochs to run before updating the target network
                            eps_clip: float, the clipping factor for trust region updates
                            gamma: float, the discount factor
                            lr_actor: float, the learning rate for the actor network
                            lr_critic: float, the learning rate for the critic network
    :param user_reward_fn: a function f(s,a,s') -> float
    :param plot_state_visit_distribution: boolean, whether to plot the state visit distribution
    :param plot_results: boolean, whether to plot of not
    :return: ppo_agent, reward_list, fitness_list
    """
    assert ("max_steps" in hyper_params.keys())
    assert ("lr" in hyper_params.keys())
    assert ("gamma" in hyper_params.keys())
    assert ("update_steps" in hyper_params.keys())
    assert ("num_episodes" in hyper_params.keys())
    assert ("eps_clip" in hyper_params.keys())
    assert ("K_epochs" in hyper_params.keys())
    assert ("entropy_coeff" in hyper_params.keys())

    max_steps = hyper_params["max_steps"]
    if "hungry-thirsty" in env.spec.id:
        env.update_step_limit(max_steps)


    ppo_agent, reward_list, fitness_list = train_ppo_agent(env=env,
                                                           hyper_params=hyper_params,
                                                           user_reward_fn=user_reward_fn,
                                                           plot_results=plot_results,
                                                           plot_state_visit_distribution=plot_state_visit_distribution)

    return ppo_agent, reward_list, fitness_list


def main():
    """ PPO HYPERPARAMETERS """
    max_steps = 200  # max timesteps per episode
    hyper_params = default_ppo_hyper_params

    ## commented out because CPU is faster
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # user input to select environment
    environments = {0: 'CartPole-v1',
                    1: 'hungry-thirsty-v0'}
    print("Choose an environment (0, 1, ...): " + str(environments))


    env_name = environments[int(input())]

    time_a = dt.datetime.now()
    env = gym.make(env_name, size=(4,4))
    env.reset()
    # env.reset(food_loc=(0,0), water_loc=(3,3))

    if "hungry-thirsty" in env.spec.id:
        env.update_step_limit(max_steps)

    ppo_agent, _, _ = train_ppo_agent(env=env,
                                      hyper_params=hyper_params,
                                      device=device,
                                      user_reward_fn=sparse_reward_fn)
    time_b = dt.datetime.now()
    print ("Time Diff: ", (time_b-time_a).total_seconds())

    while True:
        print("View the trained agent? Y/N")
        if input() == "y":
            run_episode(env=env, ppo_agent=ppo_agent, render=True, reward_fn=sparse_reward_fn)
        else:
            break


if __name__ == "__main__":
    main()

import time

import ipywidgets
import torch
import gym
import numpy as np

from Utils import InteractiveLearningCurvePlot
from Utils import ActorCriticNetwork
from gym_hungry_thirsty.envs.hungry_thirsty_reward_fns import *
import gym_hungry_thirsty

from tqdm import tqdm
from Utils import RolloutBuffer
import datetime as dt
from default_parameters import *

device = torch.device("cpu")

"""
Original paper (alg - S3): "Asynchronous Methods for Deep Reinforcement Learning" by Mnih et al. 
A reference codebase: https://medium.com/deeplearningmadeeasy/advantage-actor-critic-a2c-implementation-944e98616b
"""


def train_actor_and_critic(optimizer, memory, q_val, gamma, entropy_coeff):
    """
    Train the actor critic network

    :param optimizer: a torch optimizer for the actor-critic network
    :param memory: a rollout buffer of the last episode
    :param q_val: float, the reward from the last r(s,a,s')
    :param gamma: float, the discount factor
    :param entropy_coeff: float, the entropy coefficient
    :return:
    """
    values = torch.stack(memory.value_estimates)
    entropies = torch.stack(memory.entropy)
    logprobs = torch.stack(memory.logprobs)
    q_vals = np.zeros((len(memory), 1))

    # target values are calculated backward; then we take the difference
    # it's super important to handle correctly done states,
    # for those cases we want our to target to be equal to the reward only (1-done computes this)
    for i, (_, _, _, _, reward, done, _) in enumerate(memory.reversed()):
        q_val = reward + gamma * q_val * (1.0 - done)
        q_vals[len(memory) - 1 - i] = q_val  # store values from the end to the beginning
    advantage = torch.Tensor(q_vals) - values

    critic_loss = advantage.pow(2).mean()
    actor_loss = (-logprobs * advantage.detach()).mean()
    loss = 0.5 * critic_loss + 1 * actor_loss - entropy_coeff * entropies.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    memory.clear()


def run_episode(env,
                actor,
                render=True,
                jupyter=False,
                canvas=None,
                reward_fn=None):
    """
    Run and visualize an episode of play

    :param env: the environment
    :param actor: a policy network
    :param render: boolean, whether to render
    :param jupyter: boolean, whether this is running in a jupyter notebook or not; jupyter means we don't have tk
    :param canvas: an ipythonwidget canvas, only to be used if in jupyter
    :param reward_fn: the reward function
    :return: float, the total undiscounted reward
    """
    state = env.reset()
    done = False
    ep_reward = 0
    ep_fitness = 0

    while not done:
        if "hungry-thirsty-v0" in env.spec.id:
            state_as_arr = env.get_state_as_np_array()
            ep_fitness += 1 if not state["hungry"] else 0

            if render and not jupyter:
                env.render(score=ep_fitness)
            elif render and jupyter:
                env.jupyter_render(canvas, score=ep_fitness)

            time.sleep(0.1)
        else:
            state_as_arr = state
            env.render()
            time.sleep(0.1)

        # select action with policy
        action, _ = actor.get_action(torch.FloatTensor(state_as_arr))
        next_state, reward, done, info = env.step(action.item())

        if reward_fn is not None:
            reward = reward_fn(state=state_as_arr, action=action.item(), new_state=next_state)

        if reward is not None:
            ep_reward += reward
        state = next_state

    env.close()
    return ep_reward, ep_fitness


def create_a2c_agent(env,
                     hyper_params,
                     user_reward_fn=None,
                     plot_state_visit_distribution=False,
                     plot_results=True):
    """
    Create and train an A2C agent

    :param env: an openai gym environment
    :param hyper_params: a dict. Must have values for all the assertions below (e.g., lr, gamma, num_episodes, ...)
    :param user_reward_fn: a function which takes state, action, and next state as parameters and returns a real number
    :param plot_state_visit_distribution: bool, whether to plot the state visit distribution or not
    :param plot_results: bool, whether to plot or not
    :return: actor_critic.actor (torch network), episode_rewards (array), episode_fitness (array)
    """
    # check all the hyper_params are in the dict
    assert ("lr" in hyper_params.keys())
    assert ("gamma" in hyper_params.keys())
    assert ("num_episodes" in hyper_params.keys())
    assert ("max_steps" in hyper_params.keys())
    assert ("plot_update_freq" in hyper_params.keys())
    assert ("neural_net_hidden_size" in hyper_params.keys())
    assert ("neural_net_extra_layers" in hyper_params.keys())
    assert ("entropy_coeff" in hyper_params.keys())
    assert ("n_step_update" in hyper_params.keys())
    lr = hyper_params["lr"]
    gamma = hyper_params["gamma"]
    num_episodes = hyper_params["num_episodes"]
    max_steps = hyper_params["max_steps"]
    plot_update_freq = hyper_params["plot_update_freq"]
    hidden_size = hyper_params["neural_net_hidden_size"]
    extra_layers = hyper_params["neural_net_extra_layers"]
    entropy_coeff = hyper_params["entropy_coeff"]
    n_step_update = hyper_params["n_step_update"]

    episode_rewards = []
    episode_fitness = []
    memory = RolloutBuffer()

    # set up the interactive plot
    if plot_results:
        plotting = {0: ("Not Hungry Count Per Episode\n" +
                        r"$\Sigma_{(s, a, s') \in \tau} \mathbb{1}(s\mathrm{[is\_hungry]=False)}$",
                        "episode_fitness", ("Episode", "Not Hungry Count"), True),
                    1: ("Undiscounted Return\n" +
                        r"Summed Reward Per Episode: $\Sigma_{(s, a, s') \in \tau} r'(s)$",
                        "episode_rewards", ("Episode", "Return"), False)}
        fig = InteractiveLearningCurvePlot(num_axes=len(plotting.items()))

    # set up state visit distribution visual
    state_history = {}
    state_dist_viewer = None

    if type(env.observation_space) is gym.spaces.discrete.Discrete:
        obs_size = env.observation_space.n
    else:
        obs_size = len(env.observation_space.high)
    action_size = env.action_space.n

    # combine networks for joint training
    actor_critic = ActorCriticNetwork(obs_size=obs_size,
                                      num_actions=action_size,
                                      device=device,
                                      hidden_size=hidden_size,
                                      extra_layers=extra_layers)
    ac_optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)

    if state_dist_viewer is None:
        state_dist_viewer = ipywidgets.Image()
        from IPython import display
        display.display(state_dist_viewer)

    # wrap training in a try-except so the user can kill the agent early and still see the results / performance
    try:
        already_plotted_dist = False
        for episode_idx in tqdm(range(num_episodes)):
            done = False
            total_reward = 0
            state = env.reset()
            steps = 0
            fitness = 0

            while not done:
                if "hungry-thirsty" in env.spec.id:
                    if not state["hungry"]:
                        fitness += 1
                    state_as_arr = env.get_state_as_np_array()

                    # plot state distribution
                    if plot_state_visit_distribution:
                        if (episode_idx % 100 == 0 and episode_idx > 0) or \
                                (episode_idx == 0 and not already_plotted_dist):
                            env.render_state_distribution(state_dist_viewer, state_visit_dict=state_history)
                            already_plotted_dist = True

                    if state['position'] in state_history.keys():
                        state_history[state['position']] += 1
                    else:
                        state_history[state['position']] = 1
                else:
                    state_as_arr = state

                action, action_log_prob, entropy = actor_critic.actor.get_action(torch.FloatTensor(state_as_arr),
                                                                                 return_entropy=True)
                next_state, reward, done, info = env.step(action.item())

                if user_reward_fn is not None:
                    reward = user_reward_fn(state=state, action=action.item(), new_state=next_state)

                total_reward += reward
                steps += 1
                memory.add_experience(action=action.item(),
                                      value_estimate=actor_critic.critic(torch.FloatTensor(state_as_arr)),
                                      state=state_as_arr,
                                      logprob=action_log_prob,
                                      entropy=entropy,
                                      reward=reward,
                                      done=done)

                state = next_state

                # train if done or num steps > max_steps
                if done or (steps % n_step_update == 0):
                    if "hungry-thirsty" in env.spec.id:
                        next_state = env.get_state_as_np_array()
                    last_q_val = actor_critic.critic(torch.FloatTensor(next_state)).detach().data.numpy()
                    train_actor_and_critic(optimizer=ac_optimizer,
                                           memory=memory,
                                           q_val=last_q_val,
                                           gamma=gamma,
                                           entropy_coeff=entropy_coeff)

            episode_rewards.append([episode_idx, [total_reward]])
            episode_fitness.append([episode_idx, [fitness]])

            # update the interactive plot
            if plot_results and episode_idx != 0 and episode_idx % plot_update_freq == 0:
                for idx in plotting.keys():
                    title, list_name, labels, draw_line = plotting[idx]
                    plotting_data = locals()[list_name]
                    if len(plotting_data) > 1:
                        fig.update_subplot(axis_id=idx,
                                           title=title,
                                           learning_performance=plotting_data,
                                           labels=labels,
                                           draw_scaling_lines=draw_line)
    except:
        return actor_critic.actor, episode_rewards, episode_fitness

    return actor_critic.actor, episode_rewards, episode_fitness


def main():
    """ A2C HYPERPARAMETERS """

    hyper_params = default_a2c_hyper_params

    # user selects env
    environments = {0: 'CartPole-v1',
                    1: 'hungry-thirsty-v0'}
    print("Choose an environment (0, 1, ...): " + str(environments))
    env_name = environments[int(input())]

    for _ in range(3):
        time_a = dt.datetime.now()
        if "hungry-thirsty" in env_name:
            env = gym.make(env_name, size=(4, 4))
        else:
            env = gym.make(env_name)
        env.reset()

        actor, _, _ = create_a2c_agent(env=env,
                                       hyper_params=hyper_params,
                                       user_reward_fn=sparse_reward_fn,
                                       plot_results=True)
        time_b = dt.datetime.now()
        print ("Time diff: ", (time_b-time_a).total_seconds())

    while True:
        print("View the trained agent? Y/N")
        if input() == "y":
            run_episode(env=env,
                        actor=actor,
                        render=True,
                        reward_fn=sparse_reward_fn)
        else:
            break


if __name__ == '__main__':
    main()

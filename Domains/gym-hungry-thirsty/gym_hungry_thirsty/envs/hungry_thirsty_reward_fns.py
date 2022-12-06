import gym_hungry_thirsty.envs.hungry_thirsty_env


def general_reward_fn(params, state, action, new_state):
    """
    The general form of a reward function for the hungry thirsty domain

    :param params: a dict containing H_and_T, N_H_and_T, H_and_N_T, N_H_and_N_T
    :param state: the starting state.
        * State is composed of a dictionary of three terms:
            {
                "position": (x,y), where 0 < x < 6; 0 < y < 6
                "hungry": boolean,
                "thirsty": boolean,
            }
    :param action: an Available_Action enum. One of: EAT, DRINK, UP, DOWN, LEFT, RIGHT
    :param new_state: the new state which the action takes you to (s,a) -> s'
        * State is composed of a dictionary of three terms:
            {
                "position": (x,y), where 0 < x < 6; 0 < y < 6
                "hungry": boolean,
                "thirsty": boolean,
            }
    :return: float
    """
    currently_hungry = new_state["hungry"]
    currently_thirsty = new_state["thirsty"]
    reward = 0.0

    if currently_hungry and currently_thirsty:
        reward = params["hungry_and_thirsty"]
    elif currently_hungry and not currently_thirsty:
        reward = params["hungry_and_not_thirsty"]
    elif not currently_hungry and currently_thirsty:
        reward = params["not_hungry_and_thirsty"]
    elif not currently_hungry and not currently_thirsty:
        reward = params["not_hungry_and_not_thirsty"]

    return reward


def adversarial_reward_fn(state, action, new_state):
    """
    An adversarial reward function for the hungry thirsty domain.

    :param state: the starting state.
        * State is composed of a dictionary of three terms:
            {
                "position": (x,y), where 0 < x < 6; 0 < y < 6
                "hungry": boolean,
                "thirsty": boolean,
            }
    :param action: an Available_Action enum. One of: EAT, DRINK, UP, DOWN, LEFT, RIGHT
    :param new_state: the new state which the action takes you to (s,a) -> s'
        * State is composed of a dictionary of three terms:
            {
                "position": (x,y), where 0 < x < 6; 0 < y < 6
                "hungry": boolean,
                "thirsty": boolean,
            }
    :return:
    """
    params = {
        "hungry_and_thirsty": -1,
        "hungry_and_not_thirsty": 0,
        "not_hungry_and_thirsty": 0.5,
        "not_hungry_and_not_thirsty": 1,
    }
    return general_reward_fn(params=params, state=state, action=action, new_state=new_state)


def sparse_reward_fn(state, action, new_state):
    """
    A sparse reward function for the hungry thirsty domain.

    :param state: the starting state.
        * State is composed of a dictionary of three terms:
            {
                "position": (x,y), where 0 < x < 6; 0 < y < 6
                "hungry": boolean,
                "thirsty": boolean,
            }
    :param action: an Available_Action enum. One of: EAT, DRINK, UP, DOWN, LEFT, RIGHT
    :param new_state: the new state which the action takes you to (s,a) -> s'
        * State is composed of a dictionary of three terms:
            {
                "position": (x,y), where 0 < x < 6; 0 < y < 6
                "hungry": boolean,
                "thirsty": boolean,
            }
    :return:
    """
    params = {
        "hungry_and_thirsty": 0,
        "hungry_and_not_thirsty": 0,
        "not_hungry_and_thirsty": 1,
        "not_hungry_and_not_thirsty": 1,
    }
    return general_reward_fn(params=params, state=state, action=action, new_state=new_state)


def nudge_reward_fn(state, action, new_state):
    """
    A nudging reward function for the hungry thirsty domain.

    :param state: the starting state.
        * State is composed of a dictionary of three terms:
            {
                "position": (x,y), where 0 < x < 6; 0 < y < 6
                "hungry": boolean,
                "thirsty": boolean,
            }
    :param action: an Available_Action enum. One of: EAT, DRINK, UP, DOWN, LEFT, RIGHT
    :param new_state: the new state which the action takes you to (s,a) -> s'
        * State is composed of a dictionary of three terms:
            {
                "position": (x,y), where 0 < x < 6; 0 < y < 6
                "hungry": boolean,
                "thirsty": boolean,
            }
    :return:
    """
    params = {
        "hungry_and_thirsty": 0.0,
        "hungry_and_not_thirsty": 0.01,
        "not_hungry_and_thirsty": 1.0,
        "not_hungry_and_not_thirsty": 1.0,
    }
    return general_reward_fn(params=params, state=state, action=action, new_state=new_state)


def wdrcf_reward_fn(state, action, new_state):
    """
    The optimal reward function for the hungry thirsty domain.

    :param state: the starting state.
        * State is composed of a dictionary of three terms:
            {
                "position": (x,y), where 0 < x < 6; 0 < y < 6
                "hungry": boolean,
                "thirsty": boolean,
            }
    :param action: an Available_Action enum. One of: EAT, DRINK, UP, DOWN, LEFT, RIGHT
    :param new_state: the new state which the action takes you to (s,a) -> s'
        * State is composed of a dictionary of three terms:
            {
                "position": (x,y), where 0 < x < 6; 0 < y < 6
                "hungry": boolean,
                "thirsty": boolean,
            }
    :return:
    """
    params = {
        "hungry_and_thirsty": -0.05,
        "hungry_and_not_thirsty": -0.01,
        "not_hungry_and_thirsty": +1.0,
        "not_hungry_and_not_thirsty": +0.5,
    }
    return general_reward_fn(params=params, state=state, action=action, new_state=new_state)

import gym_hungry_thirsty.envs.hungry_thirsty_env

def reward_fn(state, action, new_state):
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
    currently_hungry = new_state["hungry"]
    currently_thirsty = new_state["thirsty"]
    reward = 0.0

    if currently_hungry and currently_thirsty:
        reward = 0.0
    elif currently_hungry and not currently_thirsty:
        reward = +0.01
    elif not currently_hungry and currently_thirsty:
        reward = +1.00
    elif not currently_hungry and not currently_thirsty:
        reward = +1.00

    return reward
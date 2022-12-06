def fitness_fn(state, action, new_state):
    """
    The fitness function for the hungry thirsty domain.

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
    :return: int, 1 if the agent is not-hungry and 0 otherwise
    """
    hungry = new_state["hungry"]

    if not hungry:
        return 1
    return 0
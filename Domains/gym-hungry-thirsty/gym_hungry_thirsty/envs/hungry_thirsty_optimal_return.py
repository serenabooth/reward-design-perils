def return_fn(trajectory):
    """

    :param trajectory: a trajectory is a list which consists of state action pairs:
        [ (state, action), (state, action), ... ]
        * Trajectories are always fixed length
        * State is composed of a dictionary of three terms:
            {
                "position": (x,y), where 0 < x < 6; 0 < y < 6
                "hungry": boolean,
                "thirsty": boolean,
            }
            to index into the dictionary, use, for example: state["position"]
        * Action is composed of an Available_Action Enum: one of EAT, DRINK, UP, DOWN, RIGHT, LEFT
        Note that some actions may have no effect:
            if the agent tries to EAT but no food is present or the agent is thirsty;
            if the agent tries to DRINK but no water is present;
            if the agent tries to move (UP, DOWN, LEFT, RIGHT), but this takes it out of bounds or into a barrier
    :return:
    """
    return_val = 0
    for (state, action) in trajectory:
        state_position = state["position"] # a tuple of (x,y), where 0 < x,y < 6
        hungry = state["hungry"] # a boolean
        thirsty = state["thirsty"] # a boolean
        action = action # an Available_Action enum: one of EAT, DRINK,

        if hungry and thirsty:
            return_val += -0.05
        elif hungry and not thirsty:
            return_val += -0.01
        elif not hungry and thirsty:
            return_val += +1.00
        elif not hungry and not thirsty:
            return_val += +0.50

    return return_val
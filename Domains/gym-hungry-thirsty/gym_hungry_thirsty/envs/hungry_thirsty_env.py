import colorsys
import itertools

import gym
from gym import spaces
import numpy as np
import random
from enum import Enum
from copy import deepcopy

#
import time
# from gym_hungry_thirsty.envs.hungry_thirsty_optimal_reward import reward_fn
import os
import ipywidgets
import numpy as np
import PIL
from io import BytesIO
import time

from matplotlib import pyplot as plt

# from scipy.ndimage import gaussian_filter

try:
    from tkinter import *
    from PIL import ImageTk
except ImportError:
    pass

try:
    import Image
except ImportError:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter

"""
A gridworld "hungry-thirsty" environment.
Originally described in this paper:

    Singh, Satinder, Richard L. Lewis, and Andrew G. Barto.
    "Where do rewards come from." Proceedings of the annual
    conference of the cognitive science society.
    Cognitive Science Society, 2009.

    Link: https://all.cs.umass.edu/pubs/2009/singh_l_b_09.pdf

"""

# debugging params
VERBOSE = False


def lookup_location(loc, size=(6, 6)):
    if loc == (0, 0):
        return "Top left"
    if loc == (0, size[1]):
        return "Bottom left"
    if loc == (size[0], size[1]):
        return "Bottom right"
    if loc == (size[0], 0):
        return "Top right"
    return "N/A - location parameter was not formatted as expected"


class Available_Actions(Enum):
    """
    The available actions for an agent to take. From the paper:
        "In addition to the movement actions, the agent has two special actions available: eat, which
        has no effect unless the agent is at the food location, where
        it causes the agent to consume food, and drink, which has
        no effect unless the the agent is at the water location, where it
        causes the agent to consume water."
    """
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    EAT = 4
    DRINK = 5

    @staticmethod
    def movement_actions():
        """
        Choose a random action from the set of available actions
        :return: Available_Actions Enum
        """
        return [Available_Actions.UP, Available_Actions.DOWN, Available_Actions.LEFT, Available_Actions.RIGHT]

    @staticmethod
    def random():
        """
        Choose a random action from the set of available actions
        :return: Available_Actions Enum
        """
        return random.choice(list(Available_Actions))

    @staticmethod
    def n():
        """
        Get the number of available actions
        :return: int (the number of available actions)
        """
        return len(list(Available_Actions))

    @staticmethod
    def all_actions():
        """
        Return all actions
        :return: Available_Actions Enum
        """
        return list(Available_Actions)


def get_random_action():
    """
    Get a random action from the Available_Actions

    :return: a random available_action (value)
    """
    return Available_Actions.random().value


def add_state_noise(input):
    """
    add noise to state

    :param input:
    :return:
    """
    return input + np.random.normal(0, 0.05, size=len(input))


def compute_reward(reward_fn_params, state):
    """
    Given a state and reward function parameters, return the reward

    :param reward_fn_params:
        either
            a 4-tuple or list (a,b,c,d), where a,b,c,d are floats
                a - hungry and thirsty reward;
                b - hungry and not thirsty reward;
                c - not hungry and thirsty reward;
                d - not hungry and not thirsty reward
        OR
            a dict with keys ["hungry and thirsty", "not hungry and thirsty",
                              "hungry and not thirsty", "not hungry and not thirsty"]
    :param state: a dictionary state of the form:
        {"position": (x,y),
         "hungry": bool,
         "thirsty": bool}
    :return: float - corresponding to the reward
    """
    if type(reward_fn_params) is list or type(reward_fn_params) is tuple:
        assert (len(reward_fn_params) == 4)

        if state["hungry"] and state["thirsty"]:
            return reward_fn_params[0]
        if state["hungry"] and not state["thirsty"]:
            return reward_fn_params[1]
        if not state["hungry"] and state["thirsty"]:
            return reward_fn_params[2]
        if not state["hungry"] and not state["thirsty"]:
            return reward_fn_params[3]

    if type(reward_fn_params is dict):
        assert ("hungry and thirsty" in reward_fn_params.keys())
        assert ("hungry and not thirsty" in reward_fn_params.keys())
        assert ("not hungry and thirsty" in reward_fn_params.keys())
        assert ("not hungry and not thirsty" in reward_fn_params.keys())

        if state["hungry"] and state["thirsty"]:
            return reward_fn_params["hungry and thirsty"]
        if state["hungry"] and not state["thirsty"]:
            return reward_fn_params["hungry and not thirsty"]
        if not state["hungry"] and state["thirsty"]:
            return reward_fn_params["not hungry and thirsty"]
        if not state["hungry"] and not state["thirsty"]:
            return reward_fn_params["not hungry and not thirsty"]

    raise Exception("Reward computation failed")


class HungryThirstyEnv(gym.Env):
    """
    Custom Environment that follows gym interface;
    Hungry-Thirsty Domain as described in the "Where do rewards come from?" paper.
    """

    def __init__(self, size=(6, 6)):
        """
        Initialize the environment
        """
        super(HungryThirstyEnv, self).__init__()

        self.action_space = spaces.Discrete(Available_Actions.n())
        high = np.array([*np.ones(shape=(size[0],size[1]), dtype=np.int64).flatten(), 1, 1], dtype=np.int64)
        low  = np.array([*np.zeros(shape=(size[0],size[1]), dtype=np.int64).flatten(), 1, 1], dtype=np.int64)
        self.observation_space = spaces.Box(high=high, low=low)

        self.last_position = None
        self.state = {"position": None,
                      "hungry": None,
                      "thirsty": None}
        self.food_loc = None
        self.water_loc = None

        self.step_ct = 0

        self.canvas_root = None
        self.render_tracking = {}  # store everything related to the canvas in this dict

        self.step_limit = 200

        self.size = size

        # visualization params
        self.GRID_WIDTH = size[0]
        self.GRID_HEIGHT = size[1]
        self.ICON_WIDTH = 50
        self.ICON_HEIGHT = 50
        self.CANVAS_WIDTH = 600
        self.CANVAS_HEIGHT = 600
        self.OFFSET = 100
        self.SHOW_GRIDLINES = True
        self.CELL_WIDTH = (self.CANVAS_WIDTH - 2 * self.OFFSET) / self.GRID_WIDTH
        self.CELL_HEIGHT = (self.CANVAS_HEIGHT - 2 * self.OFFSET) / self.GRID_HEIGHT

        home_dir = os.path.expanduser('~')
        self.FOOD_IMG_IPY = ipywidgets.Image.from_file(home_dir + "/reward-design/VisualAssets/PNG/soup_small.png")
        self.WATER_IMG_IPY = ipywidgets.Image.from_file(home_dir + "/reward-design/VisualAssets/PNG/water_small.png")
        self.AGENT_IMG_IPY = ipywidgets.Image.from_file(home_dir + "/reward-design/VisualAssets/PNG/cute_small.png")

        self.FOOD_IMG = Image.open(home_dir + "/reward-design/VisualAssets/PNG/icons8-soup-plate-100.png")
        self.WATER_IMG = Image.open(home_dir + "/reward-design/VisualAssets/PNG/icons8-water-100.png")
        self.AGENT_IMG = Image.open(home_dir + "/reward-design/VisualAssets/PNG/icons8-cute-100.png")

        self.FOOD_IMG = self.FOOD_IMG.resize((self.ICON_WIDTH, self.ICON_HEIGHT), Image.ANTIALIAS)
        self.WATER_IMG = self.WATER_IMG.resize((self.ICON_WIDTH, self.ICON_HEIGHT), Image.ANTIALIAS)
        self.AGENT_IMG = self.AGENT_IMG.resize((self.ICON_WIDTH, self.ICON_HEIGHT), Image.ANTIALIAS)

        # Read the banned transitions off of figure 3 in the paper (e.g., cannot move up from state(0,3) to state(0,2)).
        self.BANNED_TRANSITIONS = None
        if size == (6, 6):
            self.BANNED_TRANSITIONS = {str(Available_Actions.UP): [(0, 3), (1, 3), (2, 3), (3, 3), (4, 3)],
                                       str(Available_Actions.DOWN): [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)],
                                       str(Available_Actions.LEFT): [(3, 1), (3, 2), (3, 3), (3, 5)],
                                       str(Available_Actions.RIGHT): [(2, 1), (2, 2), (2, 3), (2, 5)]}
            self.BLOCKED_LINES = [[self.OFFSET,
                                   self.CANVAS_HEIGHT / 2,
                                   self.CANVAS_WIDTH - self.OFFSET - self.CELL_WIDTH,
                                   self.CANVAS_HEIGHT / 2],
                                  [self.CANVAS_WIDTH / 2,
                                   self.CELL_HEIGHT + self.OFFSET,
                                   self.CANVAS_WIDTH / 2,
                                   self.CANVAS_HEIGHT - self.OFFSET - 2 * self.CELL_HEIGHT],
                                  [self.CANVAS_WIDTH / 2,
                                   5 * self.CELL_HEIGHT + self.OFFSET,
                                   self.CANVAS_WIDTH / 2,
                                   self.CANVAS_HEIGHT - self.OFFSET]]
        elif size == (5, 5):
            self.BANNED_TRANSITIONS = {str(Available_Actions.UP): [(0, 3), (1, 3), (2, 3), (3, 3)],
                                       str(Available_Actions.DOWN): [(0, 2), (1, 2), (2, 2), (3, 2)],
                                       str(Available_Actions.LEFT): [(3, 1), (3, 2), (3, 4)],
                                       str(Available_Actions.RIGHT): [(2, 1), (2, 2), (2, 4)]}
            self.BLOCKED_LINES = [[self.OFFSET,
                                   self.CELL_HEIGHT * 3 + self.OFFSET,
                                   self.CANVAS_WIDTH - self.OFFSET - self.CELL_WIDTH,
                                   self.CELL_HEIGHT * 3 + self.OFFSET],
                                  [self.CELL_WIDTH * 3 + self.OFFSET,
                                   self.CELL_HEIGHT + self.OFFSET,
                                   self.CELL_WIDTH * 3 + self.OFFSET,
                                   self.CANVAS_HEIGHT - self.OFFSET - 2 * self.CELL_HEIGHT],
                                  [self.CELL_WIDTH * 3 + self.OFFSET,
                                  4 * self.CELL_HEIGHT + self.OFFSET,
                                  self.CELL_WIDTH * 3 + self.OFFSET,
                                  self.CANVAS_HEIGHT - self.OFFSET]]

        elif size == (4, 4):
            self.BANNED_TRANSITIONS = {str(Available_Actions.UP): [(0, 2), (1, 2), (2, 2)],
                                       str(Available_Actions.DOWN): [(0, 1), (1, 1), (2, 1)],
                                       str(Available_Actions.LEFT): [(2, 1), (2, 3)],
                                       str(Available_Actions.RIGHT): [(1, 1), (1, 3)]}
            self.BLOCKED_LINES = [[self.OFFSET,
                                   self.CELL_HEIGHT * 2 + self.OFFSET,
                                   self.CANVAS_WIDTH - self.OFFSET - self.CELL_WIDTH,
                                   self.CELL_HEIGHT * 2 + self.OFFSET],
                                  [self.CELL_WIDTH * 2 + self.OFFSET,
                                   self.CELL_HEIGHT + self.OFFSET,
                                   self.CELL_WIDTH * 2 + self.OFFSET,
                                   self.CANVAS_HEIGHT - self.OFFSET - 2 * self.CELL_HEIGHT],
                                  [self.CELL_WIDTH * 2 + self.OFFSET,
                                   self.CELL_HEIGHT * 3 + self.OFFSET,
                                   self.CELL_WIDTH * 2 + self.OFFSET,
                                   self.CANVAS_HEIGHT - self.OFFSET]]

    def update_step_limit(self, step_limit):
        """

        :param step_limit: int
        :return:
        """
        self.step_limit = step_limit

    def get_random_action(self):
        """
        Get a random action value

        :return: The value of an Available Action
        """
        return Available_Actions.random().value

    def get_available_actions(self, state):
        """
        Get a random action value from the available actions
        Since any action can occur in any state (even though it might not have an effect),
        just return all the actions

        :param state: a dict corresponding to a specific state
        :return: The value of all Available Actions
        """

        return [x.value for x in list(Available_Actions)]

    def get_available_transitions_and_probs(self, state, action, reward_fn):
        """
        Given a state and an action, compute the possible next states, rewards, and associated probabilities
        for those transitions

        :param state: a string or dict, corresponding to a state
        :param action: an int or Available Action enum, corresponding to an action
        :return: a list of the form:
            [(p1, s1, r1), ...,  (pn, sn, rn)] where p1, pn are probabilities; s1, sn are states; r1, rn are rewards
        """
        # assert the types make sense (state - str or dict, action - int, np.int64, or enum)
        assert (type(state) is str or type(state) is dict)
        assert (type(action) is int or type(action) is np.int64 or type(action) is Available_Actions)
        # convert state to dict; action to enum
        if type(state) is str:
            state = self.reverse_hash(state)
        if type(action) is int or type(action) is np.int64:
            action = Available_Actions(action)

        # this is the main data structure for tracking this
        probs_and_transitions = []

        position = state["position"]
        initial_state = deepcopy(state)

        # for all movement actions, there are two possible outcomes:
        # either the agent becomes hungry (if it's not already) and remains not thirsty (0.9 prob)
        if action in Available_Actions.movement_actions():
            if position in self.BANNED_TRANSITIONS[str(action)]:
                pass  # do nothing - do not update state position
            elif action == Available_Actions.RIGHT:
                tmp_future_position = (position[0] + 1, position[1] + 0)
                if not self.check_out_of_bounds(tmp_future_position=tmp_future_position):
                    state["position"] = tmp_future_position
            elif action == Available_Actions.LEFT:
                tmp_future_position = (position[0] - 1, position[1] + 0)
                if not self.check_out_of_bounds(tmp_future_position=tmp_future_position):
                    state["position"] = tmp_future_position
            elif action == Available_Actions.DOWN:
                tmp_future_position = (position[0] + 0, position[1] + 1)
                if not self.check_out_of_bounds(tmp_future_position=tmp_future_position):
                    state["position"] = tmp_future_position
            elif action == Available_Actions.UP:
                tmp_future_position = (position[0] + 0, position[1] - 1)
                if not self.check_out_of_bounds(tmp_future_position=tmp_future_position):
                    state["position"] = tmp_future_position

            # add the 2 possible outcomes
            state["hungry"] = True
            reward = reward_fn(state=initial_state,
                               action=action,
                               new_state=state)
            probs_and_transitions.append((0.9, reward, deepcopy(state)))
            state["thirsty"] = True
            reward = reward_fn(state=initial_state,
                               action=action,
                               new_state=state)
            probs_and_transitions.append((0.1, reward, deepcopy(state)))

        # if the action is eating - either successful or not
        elif action == Available_Actions.EAT:
            if position == self.food_loc and not state["thirsty"]:
                state["hungry"] = False
            else:
                state["hungry"] = True
            reward = reward_fn(state=initial_state,
                               action=action,
                               new_state=state)
            probs_and_transitions.append((0.9, reward, deepcopy(state)))
            state["thirsty"] = True
            reward = reward_fn(state=initial_state,
                               action=action,
                               new_state=state)
            probs_and_transitions.append((0.1, reward, deepcopy(state)))

        # drinking - either successful (not thirsty) or not (0.9 prob current state, 0.1 prob thirsty)
        elif action == Available_Actions.DRINK:
            state["hungry"] = True
            if position == self.water_loc:
                state["thirsty"] = False
                reward = reward_fn(state=initial_state,
                                   action=action,
                                   new_state=state)
                probs_and_transitions.append((1, reward, deepcopy(state)))
            else:
                # add the 2 possible outcomes - stays the same or becomes thirsty
                reward = reward_fn(state=initial_state,
                                   action=action,
                                   new_state=state)
                probs_and_transitions.append((0.9, reward, deepcopy(state)))
                state["thirsty"] = True
                reward = reward_fn(state=initial_state,
                                   action=action,
                                   new_state=state)
                probs_and_transitions.append((0.1, reward, deepcopy(state)))

        return probs_and_transitions

    def hash_lookup(self, state):
        """
        concatenate state vars (x, y, hungry, and thirsty) to create a unique lookup string

        :param state:
        :return: string
        """
        x = state["position"][0]
        y = state["position"][1]
        hungry = int(state["hungry"])
        thirsty = int(state["thirsty"])
        return str(x) + "_" + str(y) + "_" + str(hungry) + "_" + str(thirsty)

    def reverse_hash(self, hashed_state):
        """
        Convert a hashed state (string) back into a state dict

        :param hashed_state:
        :return: string
        """
        indeces = [m.start() for m in re.finditer('_', hashed_state)]
        x = int(hashed_state[0:indeces[0]])
        y = int(hashed_state[indeces[0] + 1:indeces[1]])
        hungry = bool(int(hashed_state[indeces[1] + 1:indeces[2]]))
        thirsty = bool(int(hashed_state[indeces[2] + 1:]))

        state = {"position": (x, y),
                 "hungry": hungry,
                 "thirsty": thirsty}
        return state

    def get_all_positions(self):
        """
        Hard-coded; return a list of all (x,y) positions for the gridworld
        :return: an array
        """
        return list(itertools.product(range(0, self.size[0]), range(0, self.size[1])))

    def get_all_states(self):
        """
        Hard-coded; return a list of all states for the gridworld
        :return: an array
        """
        return list(itertools.product(range(0, self.size[0]), range(0, self.size[1]), range(0, 2), range(0, 2)))

    def construct_q_table(self):
        """
        Hard-coded.
        Observations are arrays:
            [x,y,hungry,thirsty] where 0 < x,y < 6 and hungry, thirsty are 1 or 0

        :return: a dict of dicts of the form:
            {
                state_0:    {action1: X, action2: Y, ...},
                state_1:    {action1: X', action2 Y', ...},
            },
            where the keys are hashed strings of the state and action values respectively
        """
        q_table_dict = {}
        for x, y, hungry, thirsty in self.get_all_states():
            state = {
                "position": (x, y),
                "hungry": hungry,
                "thirsty": thirsty
            }
            lookup_id = self.hash_lookup(state)
            q_table_dict[lookup_id] = {}
            for action in Available_Actions.all_actions():
                q_table_dict[lookup_id][action.value] = 0 #.001 * np.random.random() - 0.001

        return q_table_dict

    def construct_value_table(self):
        """
        Hard-coded.
        Observations are arrays:
            [x,y,hungry,thirsty] where 0 < x,y < 6 and hungry, thirsty are 1 or 0

        :return: a dict where keys are hashed strings of the state and values are initialized to zero
        """
        value_table_dict = {}
        for x, y, hungry, thirsty in self.get_all_states():
            state = {
                "position": (x, y),
                "hungry": hungry,
                "thirsty": thirsty
            }
            lookup_id = self.hash_lookup(state)
            value_table_dict[lookup_id] = 0


        return value_table_dict

    def check_out_of_bounds(self, tmp_future_position):
        """
        Given a tmp_position, check whether this move would be legal:
            is the location blocked?
            is the location out of bounds?

        :param tmp_future_position: tuple (x,y) corresponding to a grid location
        :return: boolean
            True if move is legal, false otherwise.
        """
        if tmp_future_position[0] < 0 or tmp_future_position[0] >= self.GRID_WIDTH or \
                tmp_future_position[1] < 0 or tmp_future_position[1] >= self.GRID_HEIGHT:
            if VERBOSE: print("Action not valid; location is out of bounds")
            return True
        else:
            return False

    def randomly_become_thirsty(self):
        """
        On each timestep, randomly become thirsty with 0.1 probability
        From paper, it's unclear if this is true immediately after drinking.

        :return: None
        """
        if random.random() < 0.1:
            self.state["thirsty"] = True

    def encode_observation(self, as_np_array=False):
        """
        Convert the state as an observation.

        TODO: might need to update this.
        :param as_np_array: boolean, indicating whether to return as np array
        :return: state, either as an np array or dictionary
        """
        if as_np_array:
            return self.get_state_as_np_array()
        return deepcopy(self.state)

    def take_action(self, action):
        """
        Take the specified action.
        If the action is a move action (up, down, left, right),
            you can only move if it doesn't take you out of bounds or across a blocked wall.
        If the action is an eat action,
            you can only eat if you're at a food location and you're not thirsty.
        If the action is a drink action,
            you can only take the action if you're at a water location.
        Otherwise, the action has no effect.

        With some probability on each step (0.1), you will become thirsty unless you just drank.

        :param action:
        :return:
        """
        assert (type(action) is Available_Actions)

        position = self.state["position"]

        # if the action is banned, do nothing
        if action in Available_Actions.movement_actions() and position in self.BANNED_TRANSITIONS[str(action)]:
            if VERBOSE:
                print("Cannot move right; blocked by a wall")
        elif action == Available_Actions.RIGHT:
            tmp_future_position = (position[0] + 1, position[1] + 0)
            if not self.check_out_of_bounds(tmp_future_position=tmp_future_position):
                self.state["position"] = tmp_future_position
        elif action == Available_Actions.LEFT:
            tmp_future_position = (position[0] - 1, position[1] + 0)
            if not self.check_out_of_bounds(tmp_future_position=tmp_future_position):
                self.state["position"] = tmp_future_position
        elif action == Available_Actions.DOWN:
            tmp_future_position = (position[0] + 0, position[1] + 1)
            if not self.check_out_of_bounds(tmp_future_position=tmp_future_position):
                self.state["position"] = tmp_future_position
        elif action == Available_Actions.UP:
            tmp_future_position = (position[0] + 0, position[1] - 1)
            if not self.check_out_of_bounds(tmp_future_position=tmp_future_position):
                self.state["position"] = tmp_future_position

        elif action == Available_Actions.EAT:
            if position == self.food_loc and not self.state["thirsty"]:
                self.state["hungry"] = False
            else:
                if VERBOSE:
                    if position != self.food_loc:
                        print("Unable to eat; not at a food location")
                    if self.state["thirsty"]:
                        print("Unable to eat; thirsty")
        elif action == Available_Actions.DRINK and position == self.water_loc:
            self.state["thirsty"] = False
        elif action == Available_Actions.DRINK and position != self.water_loc:
            if VERBOSE:
                print("Unable to drink; not at a water location")

        self.step_ct += 1

        # become hungry after 1 step if you haven't just eaten
        if not (action == Available_Actions.EAT and
                position == self.food_loc and
                not self.state["thirsty"]):
            self.state["hungry"] = True
        # if you haven't just drunk water, randomly become thirsty with some probability
        if not action == Available_Actions.DRINK or not position == self.water_loc:
            # TODO not sure if this should also apply if you *just* drank (ambiguous in the paper)
            self.randomly_become_thirsty()

    def step(self, action, state_as_np_array=False):
        """
        Take the specified action

        :param action: an Available_Action Enum or an integer
        :param state_as_np_array: boolean, indicating whether to return as np array or dict
        :return: (obs, reward, done, info)
            obs: an encoded state
            reward: a real number corresponding to r(s, a, s')
            done: boolean indicating whether the episode terminates or not
            info: anything else (currently just an empty dict)
        """
        assert (type(action) is int
                or type(action) is np.int64
                or type(action) is Available_Actions)

        # convert action to enum if needed
        if type(action) is int or type(action) is np.int64:
            action = Available_Actions(action)

        # Execute one time step within the environment
        self.take_action(action=action)

        obs = self.encode_observation(as_np_array=state_as_np_array)
        # reward fn from the paper
        reward = None  # reward_fn(hungry=self.state["hungry"],
        #        thirsty=self.state["thirsty"],
        #        position=self.state["position"])
        done = self.step_ct == self.step_limit
        info = {}

        if VERBOSE:
            print("Agent is currently hungry (" + str(self.state["hungry"]) +
                  ") and thirsty (" + str(self.state["thirsty"]) + ")")

        return obs, reward, done, info

    def reset(self, state_as_np_array=False, food_loc=None, water_loc=None, initial_state=None,
              new_water_food_loc=False):
        """
        reset the environment: canvas, food_loc, water_loc, state, etc.

        :param state_as_np_array: boolean, indicating whether to return as np array or dict
        :return: obs (current state, encoded)
        """
        # Reset the state of the environment
        if self.render_tracking:
            self.canvas_root.destroy()
            self.render_tracking = {}
            self.canvas_root = None

        possible_food_water_locations = [(0, 0),
                                         (0, self.GRID_HEIGHT - 1),
                                         (self.GRID_WIDTH - 1, 0),
                                         (self.GRID_WIDTH - 1, self.GRID_HEIGHT - 1)]
        random.shuffle(possible_food_water_locations)

        # if self.food_loc is already defined, don't change it
        if (self.food_loc is None and food_loc is None) or new_water_food_loc:
            self.food_loc = possible_food_water_locations.pop(0)
        elif food_loc is not None:
            self.food_loc = food_loc

        # if self.water_loc is already defined, don't change it
        if (self.water_loc is None and water_loc is None) or new_water_food_loc:
            self.water_loc = possible_food_water_locations.pop(0)
        elif water_loc is not None:
            self.water_loc = water_loc

        if initial_state is None:
            self.state = {"position": (random.randint(0, self.GRID_WIDTH - 1), random.randint(0, self.GRID_HEIGHT - 1)),
                          "hungry": False,
                          "thirsty": False}
        else:
            if "position" in initial_state.keys() and \
                    "hungry" in initial_state.keys() and \
                    "thirsty" in initial_state.keys():
                self.state = initial_state
            else:
                raise Exception("State variable is improperly formatted")

        self.last_position = self.state["position"]
        self.step_ct = 0

        return self.encode_observation(as_np_array=state_as_np_array)

    def get_state_as_np_array(self, add_noise=True):
        """
        Convert the current state to an np array for learning

        :param add_noise: boolean
        :return: np.array of [x, y, hungry, thirsty]
        """
        state_arr = np.zeros(shape=(self.size[0],self.size[1]), dtype=np.int64)
        x = self.state["position"][0]
        y = self.state["position"][1]
        state_arr[x][y] = 1
        hungry = 1 if self.state["hungry"] else 0
        thirsty = 1 if self.state["thirsty"] else 0

        state_as_arr = np.array([*state_arr.flatten(), hungry, thirsty], dtype=np.int64)

        # state_as_arr = np.array([x, y, hungry, thirsty])
        if add_noise:
            return add_state_noise(input=state_as_arr)

        return state_as_arr

    def render_state_distribution(self, wIm, state_visit_dict):
        """

        :param wIm: ipywidgets Image
        :param state_visit_dict:
        :return:
        """
        # Make an RGBA array for the image
        background = np.zeros((self.CANVAS_HEIGHT, self.CANVAS_WIDTH, 4), dtype=np.uint8)
        background[:, :, 3] = 255  # opacity
        background[:, :, 0:3] = 255  # color white

        # show gridlines
        if self.SHOW_GRIDLINES:
            # vertical lines
            for i in range(0, self.GRID_WIDTH + 1):
                background[self.OFFSET:self.CANVAS_WIDTH - self.OFFSET,
                           int(i * self.CELL_HEIGHT + self.OFFSET), 0:3] = 0
            # horizontal lines
            for i in range(0, self.GRID_HEIGHT + 1):
                background[int(i * self.CELL_WIDTH + self.OFFSET),
                           self.OFFSET:self.CANVAS_HEIGHT - self.OFFSET, 0:3] = 0

        # show blocked lines (impassable)
        for i in range(len(self.BLOCKED_LINES)):
            x0, y0, x1, y1 = [int(l) for l in self.BLOCKED_LINES[i]]
            if x0 == x1:
                x0 -= 5
                x1 += 5
            if y0 == y1:
                y0 -= 5
                y1 += 5
            background[y0:y1, x0:x1, 0:1] = 255
            background[y0:y1, x0:x1, 1:3] = 0

        # add the food and water on top
        backgroundIm = Image.fromarray(background, mode="RGBA")
        food_x = int(self.food_loc[0] * self.CELL_WIDTH +
                     int(self.CELL_WIDTH / 2) - int(self.ICON_WIDTH / 2) + self.OFFSET)
        food_y = int(self.food_loc[1] * self.CELL_HEIGHT +
                     int(self.CELL_HEIGHT / 2) - int(self.ICON_HEIGHT / 2) + self.OFFSET)
        backgroundIm.alpha_composite(self.FOOD_IMG, (food_x, food_y))

        water_x = int(self.water_loc[0] * self.CELL_WIDTH +
                      int(self.CELL_WIDTH / 2) - int(self.ICON_WIDTH / 2) + self.OFFSET)
        water_y = int(self.water_loc[1] * self.CELL_HEIGHT +
                      int(self.CELL_HEIGHT / 2) - int(self.ICON_HEIGHT / 2) + self.OFFSET)
        backgroundIm.alpha_composite(self.WATER_IMG, (water_x, water_y))

        # construct a heatmap
        heat_map = np.zeros((self.CANVAS_HEIGHT, self.CANVAS_WIDTH, 4), dtype=np.uint8)
        heat_map[:, :, 3] = 255  # opacity
        heat_map[:, :, 0:3] = 255  # color white
        heat_max, heat_min = 0, 0
        if state_visit_dict:
            heat_max = max(state_visit_dict.values())

        for position in self.get_all_positions():
            y_0 = int(position[0] * self.CELL_WIDTH + self.OFFSET)
            y_1 = int(position[0] * self.CELL_WIDTH + self.OFFSET + self.CELL_WIDTH)

            x_0 = int(position[1] * self.CELL_HEIGHT + self.OFFSET)
            x_1 = int(position[1] * self.CELL_HEIGHT + self.OFFSET + self.CELL_HEIGHT)

            if position in state_visit_dict.keys():
                value = state_visit_dict[position] / heat_max
            else:
                value = 0

            heat_map[x_0:x_1, y_0:y_1, :] = np.array(plt.get_cmap('Reds')(value)) * 255

        heatmapIm = Image.fromarray(heat_map, mode="RGBA")
        heatmapIm = heatmapIm.filter(ImageFilter.GaussianBlur(radius=10))

        # heat_map = gaussian_filter(heat_map, sigma=5, multichannel=True)

        with BytesIO() as fOut:
            composite = Image.blend(heatmapIm, backgroundIm, alpha=0.3)
            composite.save(fOut, format="png")
            byPng = fOut.getvalue()

        wIm.value = byPng

    def jupyter_render(self, wIm, score=None):
        """
        Alternate rendering code for jupyter, since Tkinter is not compatible

        :param wIm: ipywidgets Image
        :param score: optional, corresponds to fitness
        :return:
        """
        # Make an RGBA array for the image
        g3 = np.zeros((self.CANVAS_HEIGHT, self.CANVAS_WIDTH, 4), dtype=np.uint8)
        g3[:, :, 3] = 255  # opacity
        g3[:, :, 0:3] = 255  # color black
        if self.SHOW_GRIDLINES:
            for i in range(0, self.GRID_WIDTH + 1):
                g3[self.OFFSET:self.CANVAS_WIDTH - self.OFFSET, int(i * self.CELL_HEIGHT + self.OFFSET), 0:3] = 0
            for i in range(0, self.GRID_HEIGHT + 1):
                g3[int(i * self.CELL_WIDTH + self.OFFSET), self.OFFSET:self.CANVAS_HEIGHT - self.OFFSET, 0:3] = 0

        for i in range(len(self.BLOCKED_LINES)):
            x0, y0, x1, y1 = [int(l) for l in self.BLOCKED_LINES[i]]
            if x0 == x1:
                x0 -= 5
                x1 += 5
            if y0 == y1:
                y0 -= 5
                y1 += 5
            g3[y0:y1, x0:x1, 0:1] = 255
            g3[y0:y1, x0:x1, 1:3] = 0

        pilIm = PIL.Image.fromarray(g3, mode="RGBA")

        food_x = int(self.food_loc[0] * self.CELL_WIDTH +
                     int(self.CELL_WIDTH / 2) - int(self.ICON_WIDTH / 2) + self.OFFSET)
        food_y = int(self.food_loc[1] * self.CELL_HEIGHT +
                     int(self.CELL_HEIGHT / 2) - int(self.ICON_HEIGHT / 2) + self.OFFSET)
        pilIm.alpha_composite(self.FOOD_IMG, (food_x, food_y))

        water_x = int(self.water_loc[0] * self.CELL_WIDTH +
                      int(self.CELL_WIDTH / 2) - int(self.ICON_WIDTH / 2) + self.OFFSET)
        water_y = int(self.water_loc[1] * self.CELL_HEIGHT +
                      int(self.CELL_HEIGHT / 2) - int(self.ICON_HEIGHT / 2) + self.OFFSET)
        pilIm.alpha_composite(self.WATER_IMG, (water_x, water_y))

        agent_x = int(self.state["position"][0] * self.CELL_WIDTH +
                      int(self.CELL_WIDTH / 2) - int(self.ICON_WIDTH / 2) + self.OFFSET)
        agent_y = int(self.state["position"][1] * self.CELL_HEIGHT +
                      int(self.CELL_HEIGHT / 2) - int(self.ICON_HEIGHT / 2) + self.OFFSET)
        pilIm.alpha_composite(self.AGENT_IMG, (agent_x, agent_y))

        hungry_text = " not " if not self.state["hungry"] else " "
        thirsty_text = " not " if not self.state["thirsty"] else " "
        agent_state_text = '"' + "I am" + hungry_text + "hungry and" + thirsty_text + "thirsty." + '"'

        img_draw = ImageDraw.Draw(pilIm)
        font = ImageFont.truetype("arial.ttf", 24)

        img_draw.text((self.OFFSET, self.CANVAS_HEIGHT - 2 * self.OFFSET / 3),
                      agent_state_text,
                      fill='black',
                      font=font)
        img_draw.text((self.OFFSET / 2, self.OFFSET / 2),
                      "Step: " + str(self.step_ct),
                      fill='black',
                      font=font)

        if score is not None:
            img_draw.text((self.CANVAS_WIDTH - 9 / 5 * self.OFFSET, self.OFFSET / 2),
                          "Score: " + str(score),
                          fill='black',
                          font=font)

        with BytesIO() as fOut:
            pilIm.save(fOut, format="png")
            byPng = fOut.getvalue()

        wIm.value = byPng

    def render(self, mode=None, score=None):
        """
        Render the gridworld.

        :param mode: used to make the function signature match OpenAI Gym spec
        :param score: int, optional
        :return: None
        """

        # if the canvas_root object doesn't exist yet, initialize it.
        if self.canvas_root is None:
            # Create an instance of tkinter frame
            self.canvas_root = Tk()

        if not self.render_tracking:
            # create canvas (background)
            self.render_tracking["canvas"] = Canvas(self.canvas_root, width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT,
                                                    bg="white")
            self.render_tracking["canvas"].pack()

            # add grids if desired
            if self.SHOW_GRIDLINES:
                for i in range(0, self.GRID_WIDTH + 1):
                    self.render_tracking["width_line_" + str(i)] = \
                        self.render_tracking["canvas"].create_line(self.OFFSET,
                                                                   i * self.CELL_HEIGHT + self.OFFSET,
                                                                   self.CANVAS_WIDTH - self.OFFSET,
                                                                   i * self.CELL_HEIGHT + self.OFFSET)
                for i in range(0, self.GRID_HEIGHT + 1):
                    self.render_tracking["height_line_" + str(i)] = self.render_tracking["canvas"].create_line(
                        i * self.CELL_WIDTH + self.OFFSET,
                        self.OFFSET,
                        i * self.CELL_WIDTH + self.OFFSET,
                        self.CANVAS_HEIGHT - self.OFFSET)

            # add blocked lines - hardcoded from paper
            for i in range(len(self.BLOCKED_LINES)):
                self.render_tracking["blocked_line_" + str(i)] = self.render_tracking["canvas"].create_line(
                    *self.BLOCKED_LINES[i],
                    width=10,
                    fill="red")
            # add food icon
            self.render_tracking["food_photoImg"] = ImageTk.PhotoImage(self.FOOD_IMG)
            self.render_tracking["food_obj"] = self.render_tracking["canvas"].create_image(
                self.food_loc[0] * self.CELL_WIDTH +
                int(self.CELL_WIDTH / 2) - int(self.FOOD_IMG.width / 2) + self.OFFSET,
                self.food_loc[1] * self.CELL_HEIGHT +
                int(self.CELL_HEIGHT / 2) - int(self.FOOD_IMG.height / 2) + self.OFFSET,
                anchor=NW,
                image=self.render_tracking["food_photoImg"])

            # add water VisualAssets
            self.render_tracking["water_photoImg"] = ImageTk.PhotoImage(self.WATER_IMG)
            self.render_tracking["water_obj"] = self.render_tracking["canvas"].create_image(
                self.water_loc[0] * self.CELL_WIDTH +
                int(self.CELL_WIDTH / 2) - int(self.WATER_IMG.width / 2) + self.OFFSET,
                self.water_loc[1] * self.CELL_HEIGHT +
                int(self.CELL_HEIGHT / 2) - int(self.WATER_IMG.height / 2) + self.OFFSET,
                anchor=NW,
                image=self.render_tracking["water_photoImg"])

            # add agent icon
            self.render_tracking["agent_photoImg"] = ImageTk.PhotoImage(self.AGENT_IMG)
            self.render_tracking["agent_obj"] = self.render_tracking["canvas"].create_image(
                self.state["position"][0] * self.CELL_WIDTH +
                int(self.CELL_WIDTH / 2) - int(self.AGENT_IMG.width / 2) + self.OFFSET,
                self.state["position"][1] * self.CELL_HEIGHT +
                int(self.CELL_HEIGHT / 2) - int(self.AGENT_IMG.height / 2) + self.OFFSET,
                anchor=NW,
                image=self.render_tracking["agent_photoImg"])

            self.render_tracking["agent_state_text"] = self.render_tracking["canvas"].create_text(
                self.CANVAS_WIDTH / 2,
                self.CANVAS_HEIGHT - self.OFFSET / 4,
                fill="black",
                font="Times 20",
                text='"N/A."')

            self.render_tracking["step_text"] = self.render_tracking["canvas"].create_text(
                self.OFFSET * 3 / 5,
                self.OFFSET / 4,
                fill="black",
                font="Times 20",
                text='"N/A."')

            self.render_tracking["score_text"] = self.render_tracking["canvas"].create_text(
                self.CANVAS_WIDTH - self.OFFSET * 9 / 5,
                self.OFFSET / 4,
                fill="black",
                font="Times 20",
                text='""')

        # reconfigure the scene as necessary
        if self.last_position != self.state["position"]:
            position = self.state["position"]
            self.render_tracking["canvas"].move(self.render_tracking["agent_obj"],
                                                (position[0] - self.last_position[0]) * self.CELL_WIDTH,
                                                (position[1] - self.last_position[1]) * self.CELL_HEIGHT)
            self.last_position = position

        # update the descriptive text
        hungry_text = " not " if not self.state["hungry"] else " "
        thirsty_text = " not " if not self.state["thirsty"] else " "
        agent_state_text = '"' + "I am" + hungry_text + "hungry and" + thirsty_text + "thirsty." + '"'
        self.render_tracking["canvas"].itemconfigure(self.render_tracking["agent_state_text"],
                                                     text=agent_state_text)

        self.render_tracking["canvas"].itemconfigure(self.render_tracking["step_text"],
                                                     text="Step: " + str(self.step_ct))

        if score is not None:
            self.render_tracking["canvas"].itemconfigure(self.render_tracking["score_text"],
                                                         text="Score: " + str(score))

        # update the rendering
        self.canvas_root.update_idletasks()
        self.canvas_root.update()

    def update_state(self, state):
        """
        Update the agent's state

        :param state:
        :return:
        """
        self.last_position = self.state["position"]
        self.state = state

    def playback(self, trajectory, episode_metadata):
        """
        Playback a trajectory

        :param trajectory: a list which consists of N state action pairs:
            [ (state, action), (state, action), ... ]
            * State is composed of a dictionary of three terms:
                {
                    "position": (x,y), where 0 < x < 6; 0 < y < 6
                    "hungry": boolean,
                    "thirsty": boolean,
                }
        :param episode_metadata: a dictionary containing "food_loc" and "water_loc" keys:
            {"food_loc": (x, y), where x = 0 or x = 5, and y = 0 or y = 5
             "water_loc": (x, y), where x = 0 or x = 5, and y = 0 or y = 5. food_loc != water_loc

        :return: None
            Play back video (render trajectory)
        """
        assert ("food_loc" in episode_metadata.keys())
        assert ("water_loc" in episode_metadata.keys())

        state, action = trajectory.pop(0)
        self.reset(food_loc=episode_metadata["food_loc"],
                   water_loc=episode_metadata["water_loc"],
                   initial_state=state)

        while len(trajectory) > 0:
            self.render()
            time.sleep(0.5)

            state, action = trajectory.pop(0)
            self.update_state(state)
            self.step_ct += 1

        self.close()

        print("Finished playing back")

    def close(self):
        """
        Quit the environment

        :return: None
        """
        if self.render_tracking:
            self.canvas_root.destroy()
            self.render_tracking = {}
            self.canvas_root = None

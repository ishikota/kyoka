#!/usr/local/bin/python

# Resolve path configucation
import os
import sys
import argparse

root = os.path.join(os.path.dirname(__file__), "../"*4)
src_path = os.path.join(root, "kyoka")
examples_path = os.path.join(root, "examples")
sys.path.append(root)
sys.path.append(src_path)
sys.path.append(examples_path)

try:
    import numpy as np
    from keras.models import Sequential
    from keras.layers.core import Dense
except ImportError:
    pass

import math

import examples.ticktacktoe.helper as Helper
from examples.ticktacktoe.task import TickTackToeTask
from examples.ticktacktoe.callback import TickTackToePerformanceWatcher

from kyoka.algorithm.q_learning import QLearning,\
        QLearningTabularActionValueFunction, QLearningApproxActionValueFunction
from kyoka.policy import EpsilonGreedyPolicy

class TickTackToeTablularValueFunction(QLearningTabularActionValueFunction):

    def generate_initial_table(self):
        board_state_num = 2**9
        action_num = 9
        table = [[[0 for a in range(action_num)]\
            for j in range(board_state_num)] for i in range(board_state_num)]
        return table

    def fetch_value_from_table(self, table, state, action):
        first_player_board, second_player_board = state
        move_position = int(math.log(action,2))
        Q_value = table[first_player_board][second_player_board][move_position]
        return Q_value

    def insert_value_into_table(self, table, state, action, new_value):
        first_player_board, second_player_board = state
        move_position = int(math.log(action,2))
        table[first_player_board][second_player_board][move_position] = new_value
        return table

class TickTackToeApproxActionValueFunction(QLearningApproxActionValueFunction):

    def __init__(self, task):
        super(TickTackToeApproxActionValueFunction, self).__init__()
        self.task = task

    def setup(self):
        super(TickTackToeApproxActionValueFunction, self).setup()
        self.model = self._build_linear_model()
        self.model.compile(loss="mse",  optimizer="adam")

    def _build_linear_model(self):
        model = Sequential()
        model.add(Dense(1, input_dim=18))
        return model

    def construct_features(self, state, action):
        return Helper.construct_features(self.task, state, action)

    def approx_predict_value(self, features):
        return self.model.predict_on_batch(np.array([features]))[0][0]

    def approx_backup(self, features, backup_target, alpha):
        loss = self.model.train_on_batch(np.array([features]), np.array([backup_target]))

TEST_LENGTH = 500000
TEST_GAME_COUNT = 10
TEST_INTERVAL = 50000
IS_FIRST_PLAYER = True

task = TickTackToeTask(is_first_player=IS_FIRST_PLAYER)
value_func = TickTackToeTablularValueFunction()
#value_func = TickTackToeApproxActionValueFunction(task)
policy = EpsilonGreedyPolicy()
policy.set_eps_annealing(1.0, 0.1, TEST_LENGTH)

callback = TickTackToePerformanceWatcher(TEST_INTERVAL, TEST_GAME_COUNT, IS_FIRST_PLAYER)

algorithm = QLearning()
algorithm.setup(task, policy, value_func)
algorithm.run_gpi(TEST_LENGTH, callbacks=callback)

flg = raw_input("Do you want to play with trained agent? (y/n) >> ")
if flg in ["y", "yes"]: Helper.play_with_agent(value_func)


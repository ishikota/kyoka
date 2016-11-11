#!/usr/local/bin/python

# Resolve path configucation
import os
import sys
import argparse

root = os.path.join(os.path.dirname(__file__), "../"*4)
src_path = os.path.join(root, "kyoka")
sample_path = os.path.join(root, "sample")
sys.path.append(root)
sys.path.append(src_path)
sys.path.append(sample_path)

import math

import sample.ticktacktoe.helper as Helper
from sample.ticktacktoe.task import TickTackToeTask
from sample.ticktacktoe.callback import TickTackToePerformanceWatcher

from kyoka.algorithm.q_learning import QLearning, QLearningTabularActionValueFunction
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


TEST_LENGTH = 500000
TEST_GAME_COUNT = 10
TEST_INTERVAL = 50000
IS_FIRST_PLAYER = True

task = TickTackToeTask(is_first_player=IS_FIRST_PLAYER)
value_func = TickTackToeTablularValueFunction()
policy = EpsilonGreedyPolicy()
policy.set_eps_annealing(1.0, 0.1, TEST_LENGTH)

callback = TickTackToePerformanceWatcher(TEST_INTERVAL, TEST_GAME_COUNT, IS_FIRST_PLAYER)

algorithm = QLearning()
algorithm.setup(task, policy, value_func)
algorithm.run_gpi(TEST_LENGTH, callbacks=callback)

flg = raw_input("Do you want to play with trained agent? (y/n) >> ")
if flg in ["y", "yes"]: Helper.play_with_agent(value_func)


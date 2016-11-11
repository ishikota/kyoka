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

import sample.maze.helper as Helper
from sample.maze.task import MazeTask
from sample.maze.callback import MazePerformanceWatcher

from kyoka.algorithm.montecarlo import MonteCarlo, MonteCarloTabularActionValueFunction
from kyoka.policy import EpsilonGreedyPolicy

class MazeTabularActionValueFunction(MonteCarloTabularActionValueFunction):

    def __init__(self, maze_shape):
        super(MazeTabularActionValueFunction, self).__init__()
        self.maze_shape = maze_shape

    def generate_initial_table(self):
        height, width = self.maze_shape
        action_num = 4
        return [[[0 for a in range(action_num)] for j in range(width)] for i in range(height)]

    def fetch_value_from_table(self, table, state, action):
        row, col = state
        return table[row][col][action]

    def insert_value_into_table(self, table, state, action, new_value):
        row, col = state
        table[row][col][action] = new_value


MAZE_FILE_PATH = os.path.join(os.path.dirname(__file__), "dyna.txt")

task = MazeTask()
task.read_maze(MAZE_FILE_PATH)
value_func = MazeTabularActionValueFunction(task.get_maze_shape())

TEST_LENGTH = 100

policy = EpsilonGreedyPolicy(eps=0.1)
policy.set_eps_annealing(1.0, 0.1, 50)
callbacks = [MazePerformanceWatcher()]
algorithm = MonteCarlo()
algorithm.setup(task, policy, value_func)
algorithm.run_gpi(TEST_LENGTH, callbacks=callbacks)


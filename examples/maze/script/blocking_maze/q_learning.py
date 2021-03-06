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

import examples.maze.helper as Helper
from examples.maze.task import MazeTask
from examples.maze.callback import MazePerformanceWatcher, MazeTransformer

from kyoka.algorithm.q_learning import QLearning, QLearningTabularActionValueFunction
from kyoka.policy import EpsilonGreedyPolicy

class MazeTabularActionValueFunction(QLearningTabularActionValueFunction):

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


TEST_LENGTH = 100
MAZE_FILE_PATH = os.path.join(os.path.dirname(__file__), "blocking.txt")
TRANSFORMATION_FILE_PATH = os.path.join(os.path.dirname(__file__), "blocking_transformed.txt")

task = MazeTask()
task.read_maze(MAZE_FILE_PATH)
value_func = MazeTabularActionValueFunction(task.get_maze_shape())
policy = EpsilonGreedyPolicy(eps=0.1)
#policy.set_eps_annealing(1.0, 0.1, 50)

callbacks = [MazePerformanceWatcher()]
transfomer = MazeTransformer()
transfomer.set_transformation(50, TRANSFORMATION_FILE_PATH)
callbacks.append(transfomer)

algorithm = QLearning()
algorithm.setup(task, policy, value_func)
algorithm.run_gpi(TEST_LENGTH, callbacks=callbacks)


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

import examples.maze.helper as Helper
from examples.maze.task import MazeTask
from examples.maze.callback import MazePerformanceWatcher

from kyoka.algorithm.montecarlo import MonteCarlo,\
        MonteCarloTabularActionValueFunction, MonteCarloApproxActionValueFunction
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

class MazeApproxActionValueFunction(MonteCarloApproxActionValueFunction):

    def __init__(self, task):
        super(MazeApproxActionValueFunction, self).__init__()
        self.task = task

    def setup(self):
        super(MazeApproxActionValueFunction, self).setup()
        self.model = self._build_linear_model()
        self.model.compile(loss="mse",  optimizer="adam")

    def _build_linear_model(self):
        maze_shape = self.task.get_maze_shape()
        input_dim = maze_shape[0] * maze_shape[1]
        model = Sequential()
        model.add(Dense(1, input_dim=input_dim))
        return model

    def construct_features(self, state, action):
        return Helper.construct_features(self.task, state, action)

    def approx_predict_value(self, features):
        return self.model.predict_on_batch(np.array([features]))[0][0]

    def approx_backup(self, features, backup_target, alpha):
        loss = self.model.train_on_batch(np.array([features]), np.array([backup_target]))

MAZE_FILE_PATH = os.path.join(os.path.dirname(__file__), "dyna.txt")

task = MazeTask()
task.read_maze(MAZE_FILE_PATH)
value_func = MazeTabularActionValueFunction(task.get_maze_shape())
#value_func = MazeApproxActionValueFunction(task)

TEST_LENGTH = 100

policy = EpsilonGreedyPolicy(eps=0.1)
policy.set_eps_annealing(1.0, 0.1, 50)
callbacks = [MazePerformanceWatcher()]
algorithm = MonteCarlo(gamma=0.01)
algorithm.setup(task, policy, value_func)
algorithm.run_gpi(TEST_LENGTH, callbacks=callbacks)


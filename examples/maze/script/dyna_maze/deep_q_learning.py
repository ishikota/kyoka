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

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

import sample.maze.helper as Helper
from sample.maze.task import MazeTask
from sample.maze.callback import MazePerformanceWatcher

from kyoka.algorithm.deep_q_learning import DeepQLearning, DeepQLearningApproxActionValueFunction
from kyoka.policy import EpsilonGreedyPolicy

class MazeApproxActionValueFunction(DeepQLearningApproxActionValueFunction):

    def __init__(self, task):
        super(MazeApproxActionValueFunction, self).__init__()
        self.task = task

    def initialize_network(self):
        maze_shape = self.task.get_maze_shape()
        input_dim = maze_shape[0] * maze_shape[1]
        model = Sequential()
        model.add(Dense(1, input_dim=input_dim))
        model.compile(loss="mse",  optimizer="adam")
        return model

    def deepcopy_network(self, q_network):
        q_hat_network = self.initialize_network()
        for original_layer, copy_layer in zip(q_network.layers, q_hat_network.layers):
            copy_layer.set_weights(original_layer.get_weights())
        return q_hat_network

    def predict_value_by_network(self, network, state, action):
        features = self.construct_features(state, action)
        return network.predict_on_batch(np.array([features]))[0][0]

    def backup_on_minibatch(self, q_network, backup_minibatch):
        minibatch = [(self.construct_features(state, action), target) for state, action, target in backup_minibatch]
        X = np.array([x for x, _ in minibatch])
        y = np.array([y for _, y in minibatch])
        loss = q_network.train_on_batch(X, y)

    def construct_features(self, state, action):
        return Helper.construct_features(self.task, state, action)

MAZE_FILE_PATH = os.path.join(os.path.dirname(__file__), "dyna.txt")

task = MazeTask()
task.read_maze(MAZE_FILE_PATH)
value_func = MazeApproxActionValueFunction(task)

TEST_LENGTH = 100
policy = EpsilonGreedyPolicy(eps=0.1)
policy.set_eps_annealing(1.0, 0.1, 50)
callbacks = [MazePerformanceWatcher()]
algorithm = DeepQLearning(N=100, C=100, minibatch_size=32, replay_start_size=50)
algorithm.setup(task, policy, value_func)
algorithm.run_gpi(TEST_LENGTH, callbacks=callbacks)


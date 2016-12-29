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

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

import examples.maze.helper as Helper
from examples.maze.task import MazeTask
from examples.maze.callback import MazePerformanceWatcher

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

import time
st = time.time()
TEST_NUM = 1
TEST_LENGTH = 100
TITLE = "approx_q_learning"
setup_algo = lambda : DeepQLearning(N=100, C=100, minibatch_size=32, replay_start_size=50)

# short annealing
short_result = []
for i in range(TEST_NUM):
    policy = EpsilonGreedyPolicy(eps=1.0)
    policy.set_eps_annealing(1.0, 0.1, 10)
    callbacks = [MazePerformanceWatcher()]
    algorithm = setup_algo()
    algorithm.setup(task, policy, value_func)
    algorithm.run_gpi(TEST_LENGTH, callbacks=callbacks)
    short_result.append([item[0] for item in callbacks[0].performance_log])
short_min = min([min(result) for result in short_result])
short_result = np.array(short_result).sum(axis=0) / (1.0*TEST_NUM)

# middle annealing
middle_result = []
for i in range(TEST_NUM):
    policy = EpsilonGreedyPolicy(eps=1.0)
    policy.set_eps_annealing(1.0, 0.1, 50)
    callbacks = [MazePerformanceWatcher()]
    algorithm = setup_algo()
    algorithm.setup(task, policy, value_func)
    algorithm.run_gpi(TEST_LENGTH, callbacks=callbacks)
    middle_result.append([item[0] for item in callbacks[0].performance_log])
middle_min = min([min(result) for result in middle_result])
middle_result = np.array(middle_result).sum(axis=0) / (1.0*TEST_NUM)

# long annealing
long_result = []
for i in range(TEST_NUM):
    policy = EpsilonGreedyPolicy(eps=1.0)
    policy.set_eps_annealing(1.0, 0.1, 100)
    callbacks = [MazePerformanceWatcher()]
    algorithm = setup_algo()
    algorithm.setup(task, policy, value_func)
    algorithm.run_gpi(TEST_LENGTH, callbacks=callbacks)
    long_result.append([item[0] for item in callbacks[0].performance_log])
long_min = min([min(result) for result in long_result])
long_result = np.array(long_result).sum(axis=0) / (1.0*TEST_NUM)

import matplotlib
matplotlib.use("tkAgg")
import matplotlib.pyplot as plt
from datetime import datetime
time_stamp = datetime.now().strftime('%m%d_%H_%M_%S')

plt.title(TITLE)
plt.plot(short_result, label="short annealing (10/100), min=%s" % short_result.min())
plt.plot(middle_result, label="middle annealing (50/100), min=%s" % middle_result.min())
plt.plot(long_result, label="long annealing (100/100), min=%s" % long_result.min())
plt.legend()
plt.savefig("performance_%s_%s.png" % (TITLE, time_stamp))

print "took time = %s(s)" % (time.time()-st)


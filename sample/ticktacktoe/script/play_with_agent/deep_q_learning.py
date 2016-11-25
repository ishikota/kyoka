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

import math

import sample.ticktacktoe.helper as Helper
from sample.ticktacktoe.task import TickTackToeTask
from sample.ticktacktoe.callback import TickTackToePerformanceWatcher

from kyoka.algorithm.deep_q_learning import DeepQLearning, DeepQLearningApproxActionValueFunction
from kyoka.policy import EpsilonGreedyPolicy

class TickTackToeApproxActionValueFunction(DeepQLearningApproxActionValueFunction):

    def __init__(self, task):
        super(TickTackToeApproxActionValueFunction, self).__init__()
        self.task = task

    def initialize_network(self):
        model = Sequential()
        model.add(Dense(1, input_dim=18))
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

TEST_LENGTH = 500000
TEST_GAME_COUNT = 10
TEST_INTERVAL = 1000
IS_FIRST_PLAYER = True

task = TickTackToeTask(is_first_player=IS_FIRST_PLAYER)
value_func = TickTackToeApproxActionValueFunction(task)
policy = EpsilonGreedyPolicy()
policy.set_eps_annealing(1.0, 0.1, TEST_LENGTH)

callback = TickTackToePerformanceWatcher(TEST_INTERVAL, TEST_GAME_COUNT, IS_FIRST_PLAYER)

algorithm = DeepQLearning(N=100000, C=1000, minibatch_size=32, replay_start_size=50000)
algorithm.setup(task, policy, value_func)
algorithm.run_gpi(TEST_LENGTH, callbacks=callback)

flg = raw_input("Do you want to play with trained agent? (y/n) >> ")
if flg in ["y", "yes"]: Helper.play_with_agent(value_func)


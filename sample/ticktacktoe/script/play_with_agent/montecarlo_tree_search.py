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

import sample.ticktacktoe.helper as Helper
from sample.ticktacktoe.helper import TickTackToeManualPolicy
from sample.ticktacktoe.task import TickTackToeTask

from kyoka.algorithm.montecarlo_tree_search import BaseMCTS, UCTNode, UCTEdge
from kyoka.callback import WatchIterationCount

class MinNode(UCTNode):

    def _edge_sort_key(self, edge):
        return -edge.calculate_value()

class MaxNode(UCTNode):
    pass

class MCTS(BaseMCTS):

    def generate_node_from_state(self, state):
        if self.next_is_first_player(state):
            return MaxNode(self.task, state)
        else:
            return MinNode(self.task, state)

    def next_is_first_player(self, state):
        return bin(state[0]|state[1]).count("1") % 2 == 0

SIMULATION_NUM = 1000
finish_rule = WatchIterationCount(SIMULATION_NUM)
next_is_first_player = lambda state: bin(state[0]|state[1]).count("1") % 2 == 0
def show_board(state):
    print "\n%s" % Helper.visualize_board(state)

task = TickTackToeTask(is_first_player=False)
algo = MCTS(TickTackToeTask(is_first_player=True))
human = TickTackToeManualPolicy()
state = TickTackToeTask().generate_initial_state()
show_board(state)
while not task.is_terminal_state(state):
    if next_is_first_player(state):
        action = algo.choose_action(state, finish_rule)
    else:
        action = human.choose_action(task, None, state)
    state = task.transit_state(state, action)
    show_board(state)


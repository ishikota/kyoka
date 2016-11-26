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

import examples.ticktacktoe.helper as Helper
from examples.ticktacktoe.callback import TickTackToePerfectPolicy
from examples.ticktacktoe.helper import TickTackToeManualPolicy
from examples.ticktacktoe.task import TickTackToeTask

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

TEST_NUM = 10
SIMULATION_NUM = 5000
finish_rule = WatchIterationCount(SIMULATION_NUM, verbose=0)
next_is_first_player = lambda state: bin(state[0]|state[1]).count("1") % 2 == 0
def show_board(state):
    print "\n%s" % Helper.visualize_board(state)

algo = MCTS(TickTackToeTask(is_first_player=True))
algo.set_finish_rule(finish_rule)

# Measure performance
minimax = TickTackToePerfectPolicy()
players = [algo, minimax]
game_results = [Helper.measure_performance(TickTackToeTask(True), None, players) for _ in range(TEST_NUM)]
result_count = [game_results.count(result) for result in [-1, 0, 1]]
result_rate  = [1.0 * count / len(game_results) for count in result_count]
print "VS PerfectPolicy average result: lose=%f, draw=%f, win=%f" % tuple(result_rate)

# Play with agent
flg = raw_input("Do you want to play with trained agent? (y/n) >> ")
fetch_action_value = lambda tree: [edge.calculate_value() for edge in tree.child_edges]
if flg in ["y", "yes"]:
    task = TickTackToeTask(is_first_player=False)
    human = TickTackToeManualPolicy()
    state = TickTackToeTask().generate_initial_state()
    show_board(state)
    while not task.is_terminal_state(state):
        if next_is_first_player(state):
            action = algo.choose_action(None, None, state)
            print "[Action Values] %s" % fetch_action_value(algo.last_calculated_tree)
        else:
            action = human.choose_action(task, None, state)
        state = task.transit_state(state, action)
        show_board(state)


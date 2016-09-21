#!/usr/local/bin/python

# Resolve path configucation
import os
import sys
import argparse

root = os.path.join(os.path.dirname(__file__), "../"*3)
src_path = os.path.join(root, "kyoka")
sample_path = os.path.join(root, "sample")
sys.path.append(root)
sys.path.append(src_path)
sys.path.append(sample_path)

import logging as log
log.basicConfig(format='[%(levelname)s] %(message)s', level=log.INFO)

from kyoka.policy.greedy_policy import GreedyPolicy

from sample.ticktacktoe.ticktacktoe_domain import TickTackToeDomain
from sample.ticktacktoe.ticktacktoe_table_value_function import TickTackToeTableValueFunction
from sample.ticktacktoe.ticktacktoe_helper import TickTackToeHelper
from sample.ticktacktoe.ticktacktoe_manual_policy import TickTackToeManualPolicy

SUPPORT_ALGORITHM = ["human", "montecarlo", "sarsa", "qlearning", "sarsalambda", "qlambda"]

parser = argparse.ArgumentParser(description="Setup game configutation")
parser.add_argument("--firstplayer", required=True, help=" or ".join(['"%s"' % algo for algo in SUPPORT_ALGORITHM]))
parser.add_argument("--secondplayer", required=True, help=" or ".join(['"%s"' % algo for algo in SUPPORT_ALGORITHM]))
args = parser.parse_args()
algos = [args.firstplayer, args.secondplayer]
for algo in algos:
  if algo not in SUPPORT_ALGORITHM:
    raise ValueError("unknown algorithm [%s] passed." % algo)


VALUE_FUNC_FILE_PATH = "%s_%s_ticktacktoe_value_function_data.pickle"
BASE_VALUE_FUNC_SAVE_PATH = os.path.join(os.path.dirname(__file__), VALUE_FUNC_FILE_PATH)

value_funcs = [TickTackToeTableValueFunction() for _ in range(2)]
[func.setUp() for func in value_funcs]

for idx, func, algo in zip(range(2), value_funcs, algos):
  which_player = "firstplayer" if idx==0 else "secondplayer"
  value_func_save_path = BASE_VALUE_FUNC_SAVE_PATH % (algo, which_player)
  if os.path.isfile(value_func_save_path):
    log.info("loading value function from %s" % value_func_save_path)
    func.load(value_func_save_path)
    log.info("finished loading value function")

domain = TickTackToeDomain()
gen_player_builder = lambda algo: TickTackToeManualPolicy if algo == "human" else GreedyPolicy
builders = [gen_player_builder(algo) for algo in algos]
players = [builder(domain, func) for builder, func in zip(builders, value_funcs)]

next_is_first_player = lambda state: bin(state[0]|state[1]).count("1") % 2 == 0
next_player = lambda state: players[0] if next_is_first_player(state) else players[1]
show_board = lambda state: log.info("\n" + TickTackToeHelper.visualize_board(state))

log.info("started the game (first player is agent")
state = domain.generate_initial_state()
show_board(state)
while not domain.is_terminal_state(state):
  action = next_player(state).choose_action(state)
  state = domain.transit_state(state, action)
  show_board(state)


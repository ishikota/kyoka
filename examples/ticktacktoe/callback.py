import random
import logging

from kyoka.callback import BaseCallback, BasePerformanceWatcher
from kyoka.policy import BasePolicy, GreedyPolicy
from examples.ticktacktoe.task import TickTackToeTask
from examples.ticktacktoe.helper import measure_performance

class TickTackToePerformanceWatcher(BasePerformanceWatcher):

    def __init__(self, test_interval, test_game_count, is_first_player=True):
        self.test_interval = test_interval
        self.test_game_count = test_game_count
        self.is_first_player = is_first_player

    def define_performance_test_interval(self):
        return self.test_interval

    def run_performance_test(self, task, value_function):
        players = self._setup_players(value_function)
        game_results = [measure_performance(task, value_function, players)\
                for _ in range(self.test_game_count)]
        result_count = [game_results.count(result) for result in [-1, 0, 1]]
        result_rate  = [1.0 * count / len(game_results) for count in result_count]
        return result_rate

    def define_log_message(self, iteration_count, task, value_function, test_result):
        return "VS PerfectPolicy average result: lose=%f, draw=%f, win=%f" % tuple(test_result)

    def _setup_players(self, value_function):
        tasks = [TickTackToeTask(is_first_player=is_first)\
            for is_first in [self.is_first_player, not self.is_first_player]]
        players = [GreedyPolicy(), TickTackToePerfectPolicy()]
        players = players if self.is_first_player else players[::-1]
        return players



class TickTackToePerfectPolicy(BasePolicy):

    def choose_action(self, task, value_function, state):
        actions = task.generate_possible_actions(state)
        states = [task.transit_state(state, action) for action in actions]
        values = [self.mini(task, state, -20, 20) for state in states]
        logging.debug("MiniMax calculation result [(action, score),...] => %s" % zip(actions, values))
        best_actions = [act for act, val in zip(actions, values) if val == max(values)]
        return random.choice(best_actions)


    def maxi(self, task, state, alpha, beta):
        if task.is_terminal_state(state): return task.calculate_reward(state)
        for action in task.generate_possible_actions(state):
            next_state = task.transit_state(state, action)
            score = self.mini(task, next_state, alpha, beta)
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def mini(self, task, state, alpha, beta):
        if task.is_terminal_state(state): return task.calculate_reward(state)
        best = 100
        for action in task.generate_possible_actions(state):
            next_state = task.transit_state(state, action)
            score = self.maxi(task, next_state, alpha, beta)
            if score <= alpha:
                return alpha
            if score < beta:
                beta = score
        return beta


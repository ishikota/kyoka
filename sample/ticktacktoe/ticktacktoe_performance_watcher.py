from kyoka.callback.base_performance_watcher import BasePerformanceWatcher
from kyoka.policy.greedy_policy import GreedyPolicy
from sample.ticktacktoe.ticktacktoe_domain import TickTackToeDomain
from sample.ticktacktoe.ticktacktoe_table_value_function import TickTackToeTableValueFunction
from sample.ticktacktoe.ticktacktoe_perfect_policy import TickTackToePerfectPolicy
from sample.ticktacktoe.ticktacktoe_helper import TickTackToeHelper

class TickTackToePerformanceWatcher(BasePerformanceWatcher):

  def __init__(self, test_interval, test_game_count, is_first_player=True):
    self.test_interval = test_interval
    self.test_game_count = test_game_count
    self.is_first_player = is_first_player

  def define_performance_test_interval(self):
    return self.test_interval

  def run_performance_test(self, domain, value_function):
    players = self.__setup_players(value_function)
    game_results = [TickTackToeHelper.measure_performance(domain, value_function, players)\
            for _ in range(self.test_game_count)]
    result_count = [game_results.count(result) for result in [-1, 0, 1]]
    result_rate  = [1.0 * count / len(game_results) for count in result_count]
    return result_rate

  def define_log_message(self, iteration_count, domain, value_function, test_result):
    return "VS PerfectPolicy average result: lose=%f, draw=%f, win=%f" % tuple(test_result)


  def __setup_players(self, value_function):
    domains = [TickTackToeDomain(is_first_player=is_first)\
        for is_first in [self.is_first_player, not self.is_first_player]]
    value_funcs = [value_function, TickTackToeTableValueFunction()]
    players = [GreedyPolicy(), TickTackToePerfectPolicy()]
    players = players if self.is_first_player else players[::-1]
    return players


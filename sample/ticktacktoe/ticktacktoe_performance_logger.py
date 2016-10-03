from kyoka.callback.base_callback import BaseCallback
from kyoka.policy.greedy_policy import GreedyPolicy
from sample.ticktacktoe.ticktacktoe_domain import TickTackToeDomain
from sample.ticktacktoe.ticktacktoe_table_value_function import TickTackToeTableValueFunction
from sample.ticktacktoe.ticktacktoe_perfect_policy import TickTackToePerfectPolicy
from sample.ticktacktoe.ticktacktoe_helper import TickTackToeHelper

class TickTackToePerformanceLogger(BaseCallback):

  def before_gpi_start(self, domain, value_function):
    self.game_log = []
    self.log_interval_counter = 0

  def after_update(self, iteration_count, domain, value_function, delta):
    iteration_count = iteration_count + 1
    if iteration_count % self.performance_test_interval == 0:
      players = self.__setup_players(value_function)
      game_results = [TickTackToeHelper.measure_performance(domain, players)\
        for _ in range(self.test_game_count)]
      result_count = [game_results.count(result) for result in [-1, 0, 1]]
      result_rate  = [1.0 * count / len(game_results) for count in result_count]
      self.game_log.append((iteration_count, result_rate))

  def set_performance_test_interval(self, interval):
      self.performance_test_interval = interval

  def set_test_game_count(self, count):
    self.test_game_count = count

  def set_is_first_player(self, is_first_player):
    self.is_first_player = is_first_player

  def __setup_players(self, value_function):
    domains = [TickTackToeDomain(is_first_player=is_first)\
        for is_first in [self.is_first_player, not self.is_first_player]]
    value_funcs = [value_function, TickTackToeTableValueFunction()]
    player_builders = [GreedyPolicy, TickTackToePerfectPolicy]
    players = [builder(domain, func) for builder, domain, func in\
        zip(player_builders, domains, value_funcs)]
    players = players if self.is_first_player else players[::-1]
    return players


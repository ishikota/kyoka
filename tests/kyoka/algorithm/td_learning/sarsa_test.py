from tests.base_unittest import BaseUnitTest
from kyoka.algorithm.td_learning.sarsa import Sarsa
from kyoka.algorithm.value_function.base_table_action_value_function import BaseTableActionValueFunction
from kyoka.algorithm.policy.greedy_policy import GreedyPolicy

from mock import Mock

class SarsaTest(BaseUnitTest):

  def setUp(self):
    self.algo = Sarsa(alpha=0.5, gamma=0.1)

  def test_update_value_function(self):
    domain = self.__setup_stub_domain()
    value_func = self.TestTableValueFunctionImpl()
    value_func.setUp()
    value_func.update_function(1, 2, 10)
    value_func.update_function(3, 4, 100)
    value_func.update_function(7, 8, 1000)
    self.debug()
    policy = GreedyPolicy(domain, value_func)
    delta = self.algo.update_value_function(domain, policy, value_func)
    self.eq([1, 4.5, 24.5], delta)
    expected = [(0, 1, 1), (1, 2, 14.5), (3, 4, 124.5)]
    for state, action, value in expected:
      self.eq(value, value_func.fetch_value_from_table(value_func.table, state, action))


  def __setup_stub_domain(self):
    mock_domain = Mock()
    mock_domain.generate_initial_state.return_value = 0
    mock_domain.is_terminal_state.side_effect = lambda state: state == 7
    mock_domain.transit_state.side_effect = lambda state, action: state + action
    mock_domain.generate_possible_actions.side_effect = lambda state: [state + 1]
    mock_domain.calculate_reward.side_effect = lambda state: state**2
    return mock_domain

  class TestTableValueFunctionImpl(BaseTableActionValueFunction):

    def generate_initial_table(self):
      return [[0 for j in range(50)] for i in range(8)]

    def fetch_value_from_table(self, table, state, action):
      return table[state][action]

    def update_table(self, table, state, action, new_value):
      table[state][action] = new_value
      return table


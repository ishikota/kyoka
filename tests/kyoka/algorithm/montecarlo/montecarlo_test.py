from tests.base_unittest import BaseUnitTest
from kyoka.algorithm.montecarlo.montecarlo import MonteCarlo
from kyoka.algorithm.policy.greedy_policy import GreedyPolicy
from kyoka.algorithm.value_function.base_action_value_function import BaseActionValueFunction
from kyoka.algorithm.value_function.base_table_action_value_function import BaseTableActionValueFunction

from mock import Mock
from nose.tools import raises

class MonteCarloTest(BaseUnitTest):

  def setUp(self):
    self.algo = MonteCarlo()

  def test_update_value_function(self):
    domain = self.__setup_stub_domain()
    value_func = self.TestTableValueFunctionImpl()
    value_func.setUp()
    policy = GreedyPolicy(domain, value_func)
    delta = self.algo.update_value_function(domain, policy, value_func)
    self.eq([59, 58, 49], delta)
    expected = [(0, 1, 59, 1), (1, 2, 58, 1), (3, 4, 49, 1), (0, 0, 0, 0)]
    update_counter = value_func.get_additinal_data("additinal_data_key_montecarlo_update_counter")
    for state, action, value, update_count in expected:
      self.eq(value, value_func.fetch_value_from_table(value_func.table, state, action))
      self.eq(update_count, value_func.fetch_value_from_table(update_counter, state, action))

  def test_update_value_function_twice_for_counter(self):
    domain = self.__setup_stub_domain()
    value_func = self.TestTableValueFunctionImpl()
    value_func.setUp()
    policy = GreedyPolicy(domain, value_func)

    self.algo.update_value_function(domain, policy, value_func)
    domain.calculate_reward.side_effect = lambda state: state
    delta = self.algo.update_value_function(domain, policy, value_func)

    self.eq([-24, -24, -21], delta)
    expected = [(0, 1, 35, 2), (1, 2, 34, 2), (3, 4, 28, 2), (0, 0, 0, 0)]
    update_counter = value_func.get_additinal_data("additinal_data_key_montecarlo_update_counter")
    for state, action, value, update_count in expected:
      self.eq(value, value_func.fetch_value_from_table(value_func.table, state, action))
      self.eq(update_count, value_func.fetch_value_from_table(update_counter, state, action))

  def test_save_update_counter_as_additinal_data(self):
    domain = self.__setup_stub_domain()
    value_func = self.TestTableValueFunctionImpl()
    value_func.setUp()
    policy = GreedyPolicy(domain, value_func)
    delta = self.algo.update_value_function(domain, policy, value_func)
    update_counter = value_func.get_additinal_data("additinal_data_key_montecarlo_update_counter")
    self.eq(1, update_counter[0][1])
    self.eq(1, update_counter[1][2])
    self.eq(1, update_counter[3][4])
    self.eq(0, update_counter[0][0])

  @raises(TypeError)
  def test_table_value_function_validation(self):
    value_func = Mock(spec=BaseActionValueFunction)
    self.algo.update_value_function("dummy", "dummy", value_func)


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
      return [[0 for j in range(50)] for i in range(4)]

    def fetch_value_from_table(self, table, state, action):
      return table[state][action]

    def update_table(self, table, state, action, new_value):
      table[state][action] = new_value
      return table



from tests.base_unittest import BaseUnitTest
from kyoka.algorithm.montecarlo.montecarlo import MonteCarlo
from kyoka.policy.greedy_policy import GreedyPolicy
from kyoka.value_function.base_action_value_function import BaseActionValueFunction
from kyoka.value_function.base_table_state_value_function import BaseTableStateValueFunction
from kyoka.value_function.base_table_action_value_function import BaseTableActionValueFunction

from mock import Mock
from nose.tools import raises

import os

class MonteCarloTest(BaseUnitTest):

  def setUp(self):
    self.algo = MonteCarlo()

  def tearDown(self):
    dir_path = self.__generate_tmp_dir_path()
    file_path = os.path.join(dir_path, "montecarlo_algorithm_state.pickle")
    if os.path.exists(dir_path):
      if os.path.exists(file_path):
        os.remove(file_path)
      os.rmdir(dir_path)

  def test_update_state_value_function(self):
    domain = self.__setup_stub_domain()
    value_func = self.TestTableStateValueFunctionImpl()
    policy = GreedyPolicy()
    self.algo.setUp(domain, policy, value_func)
    self.algo.update_value_function(domain, policy, value_func)
    expected = [(1, 59, 1), (3, 58, 1), (7, 49, 1), (0, 0, 0)]
    update_counter = self.algo.update_counter
    for state, value, update_count in expected:
      self.eq(value, value_func.fetch_value_from_table(value_func.table, state))
      self.eq(update_count, value_func.fetch_value_from_table(update_counter, state))

  def test_update_action_value_function(self):
    domain = self.__setup_stub_domain()
    value_func = self.TestTableActionValueFunctionImpl()
    policy = GreedyPolicy()
    self.algo.setUp(domain, policy, value_func)
    self.algo.update_value_function(domain, policy, value_func)
    expected = [(0, 1, 59, 1), (1, 2, 58, 1), (3, 4, 49, 1), (0, 0, 0, 0)]
    update_counter = self.algo.update_counter
    for state, action, value, update_count in expected:
      self.eq(value, value_func.fetch_value_from_table(value_func.table, state, action))
      self.eq(update_count, value_func.fetch_value_from_table(update_counter, state, action))

  def test_update_action_value_function_twice_for_counter(self):
    domain = self.__setup_stub_domain()
    value_func = self.TestTableActionValueFunctionImpl()
    policy = GreedyPolicy()
    self.algo.setUp(domain, policy, value_func)

    self.algo.update_value_function(domain, policy, value_func)
    domain.calculate_reward.side_effect = lambda state: state
    self.algo.update_value_function(domain, policy, value_func)

    expected = [(0, 1, 35, 2), (1, 2, 34, 2), (3, 4, 28, 2), (0, 0, 0, 0)]
    update_counter = self.algo.update_counter
    for state, action, value, update_count in expected:
      self.eq(value, value_func.fetch_value_from_table(value_func.table, state, action))
      self.eq(update_count, value_func.fetch_value_from_table(update_counter, state, action))

  def test_save_update_counter_as_additinal_data(self):
    dir_path = self.__generate_tmp_dir_path()
    file_path = os.path.join(dir_path, "montecarlo_algorithm_state.pickle")
    os.mkdir(dir_path)

    domain = self.__setup_stub_domain()
    value_func = self.TestTableActionValueFunctionImpl()
    policy = GreedyPolicy()
    self.algo.setUp(domain, policy, value_func)
    self.algo.update_value_function(domain, policy, value_func)
    self.algo.save_algorithm_state(dir_path)
    self.true(os.path.exists(file_path))

    new_algo = MonteCarlo()
    new_algo.setUp(domain, policy, value_func)
    new_algo.load_algorithm_state(dir_path)
    update_counter = new_algo.update_counter
    self.eq(1, update_counter[0][1])
    self.eq(1, update_counter[1][2])
    self.eq(1, update_counter[3][4])
    self.eq(0, update_counter[0][0])

  @raises(TypeError)
  def test_table_value_function_validation(self):
    value_func = Mock(spec=BaseActionValueFunction)
    self.algo.setUp("dummy", "dummy", value_func)

  def test_raise_error_when_load_failed(self):
    with self.assertRaises(IOError) as e:
      self.algo.load_algorithm_state("hoge")
    self.include("hoge", e.exception.message)


  def __setup_stub_domain(self):
    mock_domain = Mock()
    mock_domain.generate_initial_state.return_value = 0
    mock_domain.is_terminal_state.side_effect = lambda state: state == 7
    mock_domain.transit_state.side_effect = lambda state, action: state + action
    mock_domain.generate_possible_actions.side_effect = lambda state: [state + 1]
    mock_domain.calculate_reward.side_effect = lambda state: state**2
    return mock_domain

  def __generate_tmp_dir_path(self):
    return os.path.join(os.path.dirname(__file__), "tmp")

  class TestTableActionValueFunctionImpl(BaseTableActionValueFunction):

    def generate_initial_table(self):
      return [[0 for j in range(50)] for i in range(4)]

    def fetch_value_from_table(self, table, state, action):
      return table[state][action]

    def update_table(self, table, state, action, new_value):
      table[state][action] = new_value
      return table

  class TestTableStateValueFunctionImpl(BaseTableStateValueFunction):

    def generate_initial_table(self):
      return [0 for j in range(50)]

    def fetch_value_from_table(self, table, state):
      return table[state]

    def update_table(self, table, state, new_value):
      table[state] = new_value
      return table




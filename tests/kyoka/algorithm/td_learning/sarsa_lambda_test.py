from tests.base_unittest import BaseUnitTest
from kyoka.algorithm.td_learning.sarsa_lambda import SarsaLambda
from kyoka.algorithm.td_learning.eligibility_trace.action_eligibility_trace\
    import ActionEligibilityTrace as EligibilityTrace
from kyoka.value_function.base_table_state_value_function import BaseTableStateValueFunction
from kyoka.value_function.base_table_action_value_function import BaseTableActionValueFunction
from kyoka.policy.base_policy import BasePolicy

from mock import Mock

import os

class SarsaLambdaTest(BaseUnitTest):

  def setUp(self):
    trace = EligibilityTrace(EligibilityTrace.TYPE_ACCUMULATING,\
        discard_threshold=0.0001+0.0001, gamma=0.1, lambda_=0.1)
    self.algo = SarsaLambda(alpha=0.5, gamma=0.1, eligibility_trace=trace)

  def tearDown(self):
    dir_path = self.__generate_tmp_dir_path()
    file_path = os.path.join(dir_path, "sarsalambda_algorithm_state.pickle")
    if os.path.exists(dir_path):
      if os.path.exists(file_path):
        os.remove(file_path)
      os.rmdir(dir_path)

  def test_setUp(self):
    self.algo = SarsaLambda(alpha=0.5, gamma=0.1)
    self.eq(None, self.algo.trace)
    self.algo.setUp("dummy", "dummy", Mock(name="value_func"))
    self.neq(None, self.algo.trace)

  def test_update_value_function(self):
    domain = self.__setup_stub_domain()
    value_func = self.TestTableActionValueFunctionImpl()
    value_func.setUp()
    value_func.update_function(1, 2, 10)
    value_func.update_function(1, 3, 11)
    value_func.update_function(3, 4, 100)
    value_func.update_function(3, 5, 101)
    policy = self.NegativePolicyImple()
    self.algo.update_value_function(domain, policy, value_func)
    expected = [(0, 1, 1.045), (1, 2, 14.245), (3, 4, 74.5)]
    for state, action, value in expected:
      self.almosteq(value, value_func.fetch_value_from_table(value_func.table, state, action), 0.01)

  def test_save_and_load_eligibility_trace(self):
    dir_path = self.__generate_tmp_dir_path()
    file_path = os.path.join(dir_path, "sarsalambda_algorithm_state.pickle")
    os.mkdir(dir_path)

    domain = self.__setup_stub_domain()
    policy = self.NegativePolicyImple()
    value_func = self.TestTableActionValueFunctionImpl()
    self.algo.setUp(domain, policy, value_func)
    value_func.update_function(1, 2, 10)
    value_func.update_function(1, 3, 11)
    value_func.update_function(3, 4, 100)
    value_func.update_function(3, 5, 101)
    self.algo.update_value_function(domain, policy, value_func)
    self.algo.save_algorithm_state(dir_path)
    self.true(os.path.exists(file_path))

    new_algo = SarsaLambda(alpha=0.5, gamma=0.1)
    new_algo.setUp(domain, policy, value_func)
    new_algo.load_algorithm_state(dir_path)
    expected = { 3: { 4 : 0.01 } }
    eligibilities = new_algo.trace.get_eligibilities()
    self.eq(1, len(eligibilities))
    for state, action, eligibility in eligibilities:
      self.almosteq(expected[state][action], eligibility, 0.001)

  def test_reject_state_value_function(self):
    domain = self.__setup_stub_domain()
    value_func = self.TestTableStateValueFunctionImpl()
    policy = self.NegativePolicyImple()
    with self.assertRaises(TypeError) as e:
      self.algo.update_value_function(domain, policy, value_func)
    self.include("TD method requires you", e.exception.message)

  def __setup_stub_domain(self):
    mock_domain = Mock()
    mock_domain.generate_initial_state.return_value = 0
    mock_domain.is_terminal_state.side_effect = lambda state: state == 7
    mock_domain.transit_state.side_effect = lambda state, action: state + action
    mock_domain.generate_possible_actions.side_effect = lambda state: [] if state == 7 else [state + 1, state + 2]
    mock_domain.calculate_reward.side_effect = lambda state: state**2
    return mock_domain

  def __generate_tmp_dir_path(self):
    return os.path.join(os.path.dirname(__file__), "tmp")

  class TestTableActionValueFunctionImpl(BaseTableActionValueFunction):

    def generate_initial_table(self):
      return [[0 for j in range(50)] for i in range(8)]

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
      table[state][action] = new_value
      return table

  class NegativePolicyImple(BasePolicy):

    def choose_action(self, domain, value_function, state):
      actions = domain.generate_possible_actions(state)
      calc_Q_value = lambda state, action: value_function.calculate_value(state, action)
      Q_value_for_actions = [calc_Q_value(state, action) for action in actions]
      min_Q_value = min(Q_value_for_actions)
      Q_act_pair = zip(Q_value_for_actions, actions)
      worst_actions = [act for Q_value, act in Q_act_pair if min_Q_value == Q_value]
      return worst_actions[0]


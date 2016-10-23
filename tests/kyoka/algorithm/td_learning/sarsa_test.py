from tests.base_unittest import BaseUnitTest
from kyoka.algorithm.td_learning.sarsa import Sarsa
from kyoka.value_function.base_state_value_function import BaseStateValueFunction
from kyoka.value_function.base_table_action_value_function import BaseTableActionValueFunction
from kyoka.policy.base_policy import BasePolicy

from mock import Mock

class SarsaTest(BaseUnitTest):

  def setUp(self):
    self.algo = Sarsa(alpha=0.5, gamma=0.1)

  def test_update_value_function(self):
    domain = self.__setup_stub_domain()
    value_func = self.TestTableValueFunctionImpl()
    value_func.setUp()
    value_func.update_function(1, 2, 10)
    value_func.update_function(1, 3, 11)
    value_func.update_function(3, 4, 100)
    value_func.update_function(3, 5, 101)
    policy = self.NegativePolicyImple()
    self.algo.update_value_function(domain, policy, value_func)
    expected = [(0, 1, 1), (1, 2, 14.5), (3, 4, 74.5)]
    for state, action, value in expected:
      self.eq(value, value_func.fetch_value_from_table(value_func.table, state, action))

  def test_reject_state_value_function(self):
    value_func = Mock(spec=BaseStateValueFunction)
    with self.assertRaises(TypeError) as e:
      self.algo.update_value_function("dummy", "dummy", value_func)
    self.include("TD method requires you", e.exception.message)

  def __setup_stub_domain(self):
    mock_domain = Mock()
    mock_domain.generate_initial_state.return_value = 0
    mock_domain.is_terminal_state.side_effect = lambda state: state == 7
    mock_domain.transit_state.side_effect = lambda state, action: state + action
    mock_domain.generate_possible_actions.side_effect = lambda state: [] if state == 7 else [state + 1, state + 2]
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

  class NegativePolicyImple(BasePolicy):

    def choose_action(self, domain, value_function, state):
      actions = domain.generate_possible_actions(state)
      calc_Q_value = lambda state, action: value_function.calculate_value(state, action)
      Q_value_for_actions = [calc_Q_value(state, action) for action in actions]
      min_Q_value = min(Q_value_for_actions)
      Q_act_pair = zip(Q_value_for_actions, actions)
      worst_actions = [act for Q_value, act in Q_act_pair if min_Q_value == Q_value]
      return worst_actions[0]


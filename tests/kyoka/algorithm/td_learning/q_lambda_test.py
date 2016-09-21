from tests.base_unittest import BaseUnitTest
from kyoka.algorithm.td_learning.q_lambda import QLambda
from kyoka.algorithm.td_learning.eligibility_trace.action_eligibility_trace\
    import ActionEligibilityTrace as EligibilityTrace
from kyoka.value_function.base_table_action_value_function import BaseTableActionValueFunction
from kyoka.policy.base_policy import BasePolicy

from mock import Mock

class QLambaTest(BaseUnitTest):

  def setUp(self):
    trace = EligibilityTrace(EligibilityTrace.TYPE_ACCUMULATING,\
        discard_threshold=0.0001+0.0001, gamma=0.1, lambda_=0.1)
    self.algo = QLambda(alpha=0.5, gamma=0.1, eligibility_trace=trace)

  def test_update_value_function(self):
    domain = self.__setup_stub_domain()
    value_func = self.TestTableValueFunctionImpl()
    value_func.setUp()
    value_func.update_function(1, 2, 10)
    value_func.update_function(1, 3, 11)
    value_func.update_function(4, 5, 10)
    value_func.update_function(4, 6, 100)
    value_func.update_function(10, 11, 1000)
    value_func.update_function(10, 12, 10000)
    policy = self.CheetPolicyImple(domain, value_func)
    delta = self.algo.update_value_function(domain, policy, value_func)
    self.almosteq([-279.5, 0.075, 1.05, 5, 7.5, 500], sorted(delta), tolerance=0.0001)
    expected = [(0, 1, 1.125), (1, 3, 23.5), (4, 6, 600), (10, 11, 720.5)]
    for state, action, value in expected:
      self.eq(value, value_func.fetch_value_from_table(value_func.table, state, action))


  def __setup_stub_domain(self):
    mock_domain = Mock()
    mock_domain.generate_initial_state.return_value = 0
    mock_domain.is_terminal_state.side_effect = lambda state: state == 21
    mock_domain.transit_state.side_effect = lambda state, action: state + action
    mock_domain.generate_possible_actions.side_effect = lambda state: [] if state == 21 else [state + 1, state + 2]
    mock_domain.calculate_reward.side_effect = lambda state: state**2
    return mock_domain

  class TestTableValueFunctionImpl(BaseTableActionValueFunction):

    def generate_initial_table(self):
      return [[0 for j in range(13)] for i in range(11)]

    def fetch_value_from_table(self, table, state, action):
      table[state][action]
      return table[state][action]

    def update_table(self, table, state, action, new_value):
      table[state][action] = new_value
      return table

  class CheetPolicyImple(BasePolicy):

    def choose_action(self, state):
      state_action_map = {
          0: 1,
          1: 3,
          4: 6,
          10: 11
      }
      return state_action_map[state]


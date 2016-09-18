from tests.base_unittest import BaseUnitTest
from kyoka.algorithm.montecarlo.montecarlo import MonteCarlo
from kyoka.algorithm.policy.greedy_policy import GreedyPolicy
from kyoka.algorithm.value_function.base_action_value_function import BaseActionValueFunction

from mock import Mock

class MonteCarloTest(BaseUnitTest):

  def setUp(self):
    self.algo = MonteCarlo()

  def test_update_value_function(self):
    domain = self.__setup_stub_domain()
    value_func = self.__setup_stub_value_function()
    policy = GreedyPolicy(domain, value_func)
    self.algo.update_value_function(domain, policy, value_func)
    update_func_arg_capture = value_func.update_function.call_args_list
    expected = [(0, 1, (1, 59)), (1, 2, (2, 39)), (3, 4, (4, 42.25))]
    for expected, capture in zip(expected, update_func_arg_capture):
      self.eq(expected, capture[0])


  def __setup_stub_domain(self):
    mock_domain = Mock()
    mock_domain.generate_initial_state.return_value = 0
    mock_domain.is_terminal_state.side_effect = lambda state: state == 7
    mock_domain.transit_state.side_effect = lambda state, action: state + action
    mock_domain.generate_possible_actions.side_effect = lambda state: [state + 1]
    mock_domain.calculate_reward.side_effect = lambda state: state**2
    return mock_domain

  def __setup_stub_value_function(self):
    mock_value_func = Mock(spec=BaseActionValueFunction)
    mock_value_func.calculate_value.side_effect = lambda state, action: 0 if state==0 else (state, action*10)
    mock_value_func.deepcopy.return_value = mock_value_func
    return mock_value_func


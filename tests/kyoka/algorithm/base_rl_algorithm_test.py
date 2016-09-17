from tests.base_unittest import BaseUnitTest
from kyoka.algorithm.base_rl_algorithm import BaseRLAlgorithm
from kyoka.algorithm.policy.greedy_policy import GreedyPolicy
from kyoka.algorithm.value_function.base_action_value_function import BaseActionValueFunction

from mock import Mock

class BaseRLAlgorithmTest(BaseUnitTest):

  def setUp(self):
    self.algo = BaseRLAlgorithm()

  def test_error_msg_when_not_implement_abstract_method(self):
    self.__check_err_msg(lambda : self.algo.GPI("dummy", "dummy", "dummy"), "GPI")

  def test_gen_episode(self):
    domain = self.__setup_stub_domain()
    value_func = self.__setup_stub_value_function()
    policy = GreedyPolicy(domain, value_func)
    episode = self.algo.generate_episode(domain, policy)
    self.eq(3, len(episode))
    self.eq((0, 1, 1, 1), episode[0])
    self.eq((1, 2, 3, 9), episode[1])
    self.eq((3, 4, 7, 49), episode[2])


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
    mock_value_func.calculate_value.return_value = 0
    return mock_value_func

  def __check_err_msg(self, target_method, target_word):
    try:
      target_method()
    except NotImplementedError as e:
      self.include(target_word, str(e))
    else:
      self.fail("NotImplementedError does not occur")


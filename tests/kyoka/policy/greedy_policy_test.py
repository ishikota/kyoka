from tests.base_unittest import BaseUnitTest
from kyoka.policy.greedy_policy import GreedyPolicy
from kyoka.value_function.base_state_value_function import BaseStateValueFunction
from mock import Mock

class GreedyPolicyTest(BaseUnitTest):

  def test_choose_action_when_best_action_is_single(self):
    domain = self.__setup_domain_stub([1,2,3])
    value_func = self.__setup_value_function_stub([100, 50, 150])
    policy = GreedyPolicy()
    greedy_action = policy.choose_action(domain, value_func, state="dummy")
    self.eq(3, greedy_action)

  def test_choose_action_when_best_action_is_multiple(self):
    domain = self.__setup_domain_stub([1,2,3])
    value_func = self.__setup_value_function_stub([100, 50, 100])
    random = self.__setup_random()
    policy = GreedyPolicy(rand=random)
    greedy_action = policy.choose_action(domain, value_func, state="dummy")
    self.eq(3, greedy_action)

  def __setup_domain_stub(self, possible_actions):
    mock_domain = Mock()
    mock_domain.generate_possible_actions.return_value = possible_actions
    return mock_domain

  def __setup_value_function_stub(self, mock_return):
    mock_value_func = Mock(spec=BaseStateValueFunction)
    mock_value_func.calculate_value.side_effect = mock_return
    return mock_value_func

  def __setup_random(self):
    random = Mock()
    random.choice.side_effect = lambda ary: ary[1]
    return random


from tests.base_unittest import BaseUnitTest
from kyoka.algorithm.policy.greedy_policy import GreedyPolicy
from mock import Mock

class GreedyPolicyTest(BaseUnitTest):

  def test_choose_action_when_best_action_is_single(self):
    domain = self.__setup_domain_stub([100, 50, 150], [1,2,3])
    policy = GreedyPolicy(domain)
    greedy_action = policy.choose_action(Q="dummy", state="dummy")
    self.eq(3, greedy_action)

  def test_choose_action_when_best_action_is_multiple(self):
    domain = self.__setup_domain_stub([100, 50, 100], [1,2,3])
    random = self.__setup_random()
    policy = GreedyPolicy(domain, rand=random)
    greedy_action = policy.choose_action(Q="dummy", state="dummy")
    self.eq(3, greedy_action)

  def __setup_domain_stub(self, fetch_Q_value_return, possible_actions):
    mock_domain = Mock()
    mock_domain.fetch_Q_value.side_effect = fetch_Q_value_return
    mock_domain.generate_possible_actions.return_value = possible_actions
    return mock_domain

  def __setup_random(self):
    random = Mock()
    random.choice.side_effect = lambda ary: ary[1]
    return random


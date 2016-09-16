from tests.base_unittest import BaseUnitTest
from kyoka.algorithm.policy.epsilon_greedy_policy import EpsilonGreedyPolicy
from mock import Mock

class EpsilonGreedyPolicyTest(BaseUnitTest):

  def test_choose_action_boundary_test(self):
    eps = 0.0001
    self.eq(1, self.__choose_action_with_rand_value(0))
    self.eq(1, self.__choose_action_with_rand_value(0.1 - eps))
    self.eq(2, self.__choose_action_with_rand_value(0.1))
    self.eq(2, self.__choose_action_with_rand_value(0.2 + (1-0.3) - eps))
    self.eq(3, self.__choose_action_with_rand_value(0.2 + (1-0.3)))
    self.eq(3, self.__choose_action_with_rand_value(1-eps))

  def xtest_sampling_action(self):
    domain = self.__setup_domain_stub([100, 150, 50]*100000, [1,2,3])
    policy = EpsilonGreedyPolicy(domain, eps=0.3)
    result = [policy.choose_action(Q="dummy", state="dummy") for _ in range(100000)]
    count = (result.count(1), result.count(2), result.count(3))
    self.debug()
    # sample result => count = (10062, 79955, 9983)


  def __choose_action_with_rand_value(self, rand_val):
    domain = self.__setup_domain_stub([100, 150, 50], [1,2,3])
    random = self.__setup_random(rand_val)
    policy = EpsilonGreedyPolicy(domain, eps=0.3, rand=random)
    return policy.choose_action(Q="dummy", state="dummy")


  def __setup_domain_stub(self, fetch_Q_value_return, possible_actions):
    mock_domain = Mock()
    mock_domain.fetch_Q_value.side_effect = fetch_Q_value_return
    mock_domain.generate_possible_actions.return_value = possible_actions
    return mock_domain

  def __setup_random(self, rand_val):
    random = Mock()
    random.random.return_value = rand_val
    random.choice.side_effect = lambda ary: ary[0]
    return random

from tests.base_unittest import BaseUnitTest
from kyoka.policy.epsilon_greedy_policy import EpsilonGreedyPolicy
from kyoka.value_function.base_action_value_function import BaseActionValueFunction
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

  def test_epsilon_annealing(self):
    policy = EpsilonGreedyPolicy(eps=0.5)
    self.false(policy.do_annealing)
    policy.set_eps_annealing(initial_eps=1.0, final_eps=0.1, anneal_duration=9)
    self.true(policy.do_annealing)
    expected_eps = [(i+1)*0.1 for i in range(9)][::-1] + [0.1, 0.1]
    for eps in expected_eps:
      policy.anneal_eps()
      self.almosteq(eps, policy.eps, 0.000001)

  def xtest_sampling_action(self):
    domain = self.__setup_domain_stub([1,2,3])
    value_func = self.__setup_value_function_stub([100, 150, 50]*100000)
    policy = EpsilonGreedyPolicy(eps=0.3)
    result = [policy.choose_action(domain. value_func, state="dummy") for _ in range(100000)]
    count = (result.count(1), result.count(2), result.count(3))
    self.debug()
    # sample result => count = (10062, 79955, 9983)


  def __choose_action_with_rand_value(self, rand_val):
    domain = self.__setup_domain_stub([1,2,3])
    value_func = self.__setup_value_function_stub([100, 150, 50])
    random = self.__setup_random(rand_val)
    policy = EpsilonGreedyPolicy(eps=0.3, rand=random)
    return policy.choose_action(domain, value_func, state="dummy")


  def __setup_domain_stub(self, possible_actions):
    mock_domain = Mock()
    mock_domain.generate_possible_actions.return_value = possible_actions
    return mock_domain

  def __setup_value_function_stub(self, mock_return):
    mock_value_func = Mock(spec=BaseActionValueFunction)
    mock_value_func.calculate_value.side_effect = mock_return
    return mock_value_func

  def __setup_random(self, rand_val):
    random = Mock()
    random.random.return_value = rand_val
    random.choice.side_effect = lambda ary: ary[0]
    return random


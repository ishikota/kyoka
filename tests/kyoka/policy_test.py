from nose.tools import raises
from mock import Mock

from kyoka.policy import BasePolicy, GreedyPolicy, EpsilonGreedyPolicy
from tests.base_unittest import BaseUnitTest


class BasePolityTest(BaseUnitTest):

    def setUp(self):
        self.policy = BasePolicy()

    @raises(NotImplementedError)
    def test_choose_action(self):
        self.policy.choose_action("dummy", "dummy", "dummy")

class GreedyPolicyTest(BaseUnitTest):

    def test_choose_action_when_best_action_is_single(self):
        task = setup_task_stub([1,2,3])
        value_func = setup_value_function_stub([100, 50, 150])
        policy = GreedyPolicy()
        greedy_action = policy.choose_action(task, value_func, state="dummy")
        self.eq(3, greedy_action)

    def test_choose_action_when_best_action_is_multiple(self):
        task = setup_task_stub([1,2,3])
        value_func = setup_value_function_stub([100, 50, 100])
        random = self.setup_random()
        policy = GreedyPolicy(rand=random)
        greedy_action = policy.choose_action(task, value_func, state="dummy")
        self.eq(3, greedy_action)

    def setup_random(self):
        random = Mock()
        random.choice.side_effect = lambda ary: ary[1]
        return random

class EpsilonGreedyPolicyTest(BaseUnitTest):

    def test_choose_action_boundary_test(self):
        eps = 0.0001
        self.eq(1, self.choose_action_with_rand_value(0))
        self.eq(1, self.choose_action_with_rand_value(0.1 - eps))
        self.eq(2, self.choose_action_with_rand_value(0.1))
        self.eq(2, self.choose_action_with_rand_value(0.2 + (1-0.3) - eps))
        self.eq(3, self.choose_action_with_rand_value(0.2 + (1-0.3)))
        self.eq(3, self.choose_action_with_rand_value(1-eps))

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
        task = setup_task_stub([1,2,3])
        value_func = setup_value_function_stub([100, 150, 50]*100000)
        policy = EpsilonGreedyPolicy(eps=0.3)
        result = [policy.choose_action(task, value_func, state="dummy") for _ in range(100000)]
        count = (result.count(1), result.count(2), result.count(3))
        self.debug()
        # sample result => count = (10062, 79955, 9983)


    def choose_action_with_rand_value(self, rand_val):
        task = setup_task_stub([1,2,3])
        value_func = setup_value_function_stub([100, 150, 50])
        random = self.setup_random(rand_val)
        policy = EpsilonGreedyPolicy(eps=0.3, rand=random)
        return policy.choose_action(task, value_func, state="dummy")

    def setup_random(self, rand_val):
        random = Mock()
        random.random.return_value = rand_val
        random.choice.side_effect = lambda ary: ary[0]
        return random



def setup_task_stub(possible_actions):
    mock_task = Mock()
    mock_task.generate_possible_actions.return_value = possible_actions
    return mock_task

def setup_value_function_stub(mock_return):
    mock_value_func = Mock()
    mock_value_func.predict_value.side_effect = mock_return
    return mock_value_func


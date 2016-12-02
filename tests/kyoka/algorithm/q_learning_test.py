from mock import Mock
from nose.tools import raises

from kyoka.value_function import BaseActionValueFunction
from kyoka.algorithm.q_learning import QLearning, QLearningTabularActionValueFunction,\
        QLearningApproxActionValueFunction, validate_value_function
from tests.base_unittest import BaseUnitTest
from tests.utils import NegativePolicy


class QLearningTest(BaseUnitTest):

    def test_value_function_validation(self):
        algo = QLearning()
        validate_value_function(QLearningTabularActionValueFunction())
        validate_value_function(QLearningApproxActionValueFunction())
        with self.assertRaises(TypeError):
            algo.setup("dummy", "dummy", BaseActionValueFunction())

    def test_run_gpi_for_an_episode(self):
        algo = QLearning(alpha=0.5, gamma=0.1)
        task = setup_stub_task()
        value_func = QLearningTabularActionValueFunctionImpl()
        policy = NegativePolicy()
        algo.setup(task, policy, value_func)

        value_func.insert_value_into_table(value_func.table, 1, 2, 10)
        value_func.insert_value_into_table(value_func.table, 1, 3, 11)
        value_func.insert_value_into_table(value_func.table, 3, 4, 100)
        value_func.insert_value_into_table(value_func.table, 3, 5, 101)

        algo.run_gpi_for_an_episode(task, policy, value_func)

        expected = [(0, 1, 1.05), (1, 2, 14.55), (3, 4, 74.5)]
        for state, action, value in expected:
            self.eq(value, value_func.predict_value(state, action))

class QLearningTabularActionValueFunctionTest(BaseUnitTest):

    def test_backup(self):
        func = QLearningTabularActionValueFunctionImpl()
        func.setup()
        func.backup(state=0, action=0, backup_target=2, alpha=0.5)
        self.eq(1, func.predict_value(0, 0))
        func.backup(state=0, action=0, backup_target=2, alpha=0.5)
        self.eq(1.5, func.predict_value(0, 0))

class QLearningApproxActionValueFunctionTest(BaseUnitTest):

    def setUp(self):
        self.func = QLearningApproxActionValueFunction()

    @raises(NotImplementedError)
    def test_error_msg_when_not_implement_construct_features(self):
        self.func.construct_features("dummy", "dummy")

    @raises(NotImplementedError)
    def test_error_msg_when_not_implement_approx_predict_value(self):
        self.func.approx_predict_value("dummy")

    @raises(NotImplementedError)
    def test_error_msg_when_not_implement_approx_backup(self):
        self.func.approx_backup("dummy", "dummy", "dummy")


class QLearningTabularActionValueFunctionImpl(QLearningTabularActionValueFunction):

    def generate_initial_table(self):
        return [[0 for j in range(50)] for i in range(8)]

    def fetch_value_from_table(self, table, state, action):
        return table[state][action]

    def insert_value_into_table(self, table, state, action, new_value):
        table[state][action] = new_value


def setup_stub_task():
    mock_task = Mock()
    mock_task.generate_initial_state.return_value = 0
    mock_task.is_terminal_state.side_effect = lambda state: state == 7
    mock_task.transit_state.side_effect = lambda state, action: state + action
    mock_task.generate_possible_actions.side_effect = lambda state: [] if state == 7 else [state + 1, state + 2]
    mock_task.calculate_reward.side_effect = lambda state: state**2
    return mock_task


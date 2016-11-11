from tests.base_unittest import BaseUnitTest
from tests.utils import generate_tmp_dir_path, setup_tmp_dir, teardown_tmp_dir
from kyoka.algorithm_.montecarlo import MonteCarlo, MontCarloTabularActionValueFunction, BaseMonteCarloApproxActionValueFunction
from kyoka.value_function_ import BaseTabularActionValueFunction
from kyoka.policy_ import GreedyPolicy

from mock import Mock
from nose.tools import raises

import os

class MonteCarloTest(BaseUnitTest):

    def setUp(self):
        self.algo = MonteCarlo()

    def tearDown(self):
        cleanup_trash()

    @raises(TypeError)
    def test_value_function_validation(self):
        self.algo.setup("dummy", "dummy", BaseTabularActionValueFunction())


    def test_run_gpi_for_an_episode(self):
        task = setup_stub_task()
        value_func = MonteCarloTabularActionValueFunctionImpl()
        policy = GreedyPolicy()
        self.algo.setup(task, policy, value_func)
        self.algo.run_gpi_for_an_episode(task, policy, value_func)
        expected = [(0, 1, 59, 1), (1, 2, 58, 1), (3, 4, 49, 1), (0, 0, 0, 0)]
        update_counter = value_func.update_counter
        for state, action, value, update_count in expected:
            self.eq(value, value_func.fetch_value_from_table(value_func.table, state, action))
            self.eq(update_count, value_func.fetch_value_from_table(update_counter, state, action))

    def test_integration(self):
        task = setup_stub_task()
        value_func = MonteCarloTabularActionValueFunctionImpl()
        policy = GreedyPolicy()
        self.algo.setup(task, policy, value_func)
        self.algo.run_gpi_for_an_episode(task, policy, value_func)
        setup_tmp_dir(__file__)
        self.algo.save(generate_tmp_dir_path(__file__))

        task.calculate_reward.side_effect = lambda state: state
        new_algo = MonteCarlo()
        new_algo.setup(task, policy, value_func)
        new_algo.load(generate_tmp_dir_path(__file__))
        new_algo.run_gpi_for_an_episode(task, policy, value_func)

        expected = [(0, 1, 35, 2), (1, 2, 34, 2), (3, 4, 28, 2), (0, 0, 0, 0)]
        update_counter = value_func.update_counter
        for state, action, value, update_count in expected:
            self.eq(value, value_func.fetch_value_from_table(value_func.table, state, action))
            self.eq(update_count, value_func.fetch_value_from_table(update_counter, state, action))


class MontCarloTabularActionValueFunctionTest(BaseUnitTest):

    def setUp(self):
        self.func = MonteCarloTabularActionValueFunctionImpl()

    def tearDown(self):
        cleanup_trash()

    def test_setup(self):
        self.func.setup()
        self.eq(4, len(self.func.update_counter))

    def test_save_and_load(self):
        self.func.setup()
        self.func.update_counter[0][0] = 1
        setup_tmp_dir(__file__)
        self.func.save(generate_tmp_dir_path(__file__))
        new_func = MonteCarloTabularActionValueFunctionImpl()
        new_func.load(generate_tmp_dir_path(__file__))
        self.eq(1, new_func.update_counter[0][0])

    def test_backup(self):
        self.func.setup()
        self.func.backup(state=0, action=1, backup_target=2, alpha="dummy")
        self.func.backup(state=1, action=0, backup_target=3, alpha="dummy")
        self.func.backup(state=0, action=1, backup_target=4, alpha="dummy")
        self.eq(3, self.func.predict_value(state=0, action=1))
        self.eq(3, self.func.predict_value(state=1, action=0))
        self.eq(2, self.func.update_counter[0][1])
        self.eq(1, self.func.update_counter[1][0])

class MonteCarloApproxActionValueFunctionTest(BaseUnitTest):

    def setUp(self):
        self.func = BaseMonteCarloApproxActionValueFunction()

    @raises(NotImplementedError)
    def test_error_msg_when_not_implement_construct_features(self):
        self.func.construct_features("dummy", "dummy")

    @raises(NotImplementedError)
    def test_error_msg_when_not_implement_approx_predict_value(self):
        self.func.approx_predict_value("dummy")

    @raises(NotImplementedError)
    def test_error_msg_when_not_implement_approx_backup(self):
        self.func.approx_backup("dummy", "dummy", "dummy")


class MonteCarloTabularActionValueFunctionImpl(MontCarloTabularActionValueFunction):

    def generate_initial_table(self):
        return [[0 for i in range(50)] for i in range(4)]

    def fetch_value_from_table(self, table, state, action):
        return table[state][action]

    def insert_value_into_table(self, table, state, action, new_value):
        table[state][action] = new_value

def setup_stub_task():
    mock_task = Mock()
    mock_task.generate_initial_state.return_value = 0
    mock_task.is_terminal_state.side_effect = lambda state: state == 7
    mock_task.transit_state.side_effect = lambda state, action: state + action
    mock_task.generate_possible_actions.side_effect = lambda state: [state + 1]
    mock_task.calculate_reward.side_effect = lambda state: state**2
    return mock_task

def cleanup_trash():
    filenames = ["montecarlo_update_counter.pickle", "montecarlo_table_action_value_function_data.pickle"]
    teardown_tmp_dir(__file__, filenames)


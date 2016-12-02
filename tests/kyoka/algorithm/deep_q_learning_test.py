import os

from nose.tools import raises
from mock import patch, Mock

from kyoka.utils import pickle_data, unpickle_data
from kyoka.policy import GreedyPolicy
from kyoka.algorithm.deep_q_learning import DeepQLearning,\
        DeepQLearningApproxActionValueFunction, ExperienceReplay
from tests.base_unittest import BaseUnitTest
from tests.utils import generate_tmp_dir_path, setup_tmp_dir, teardown_tmp_dir, NegativePolicy


class DeepQLearningTest(BaseUnitTest):

    def setUp(self):
        self.algo = DeepQLearning(gamma=0.1, N=3, C=3, minibatch_size=2, replay_start_size=2)
        self.algo.replay_memory.store_transition(2.5, 3, 25, 5)
        self.algo.replay_memory.store_transition(5.0, 7, 144, 4)
        self.task = setup_stub_task()
        self.policy = NegativePolicy()
        self.value_func = self.DeepQLearningApproxActionValueFunctionImpl()
        self.algo.setup(self.task, self.policy, self.value_func)

    def tearDown(self):
        cleanup_trash()

    def test_initialize_replay_memory(self):
        algo = DeepQLearning(gamma=0.1, N=3, C=3, minibatch_size=2, replay_start_size=2)
        task = setup_stub_task()
        policy = NegativePolicy()
        value_func = self.DeepQLearningApproxActionValueFunctionImpl(strict_mode=False)
        # Overrider terminal judge logic to avoid infinite episode by random policy
        task.is_terminal_state.side_effect = lambda state: state == 4 or state >= 100
        self.eq(0, len(algo.replay_memory.queue))
        algo.setup(task, policy, value_func)
        self.eq(2, len(algo.replay_memory.queue))

    def test_check_backup_minibatch_delivery_in_gpi(self):
        with patch('random.sample', side_effect=lambda lst, n: lst[len(lst)-n:]):
            self.algo.run_gpi_for_an_episode(self.task, self.policy, self.value_func)

        backup_minibatch_expected = [
                [(5.0, 7, 144), (0, 1, 1.6)],
                [(0, 1, 1.6), (1, 3, 16)]
        ]
        actual = [arg[0][0] for arg in self.value_func.q_network.train_on_minibatch.call_args_list]
        self.eq(backup_minibatch_expected, actual)
        self.value_func.q_hat_network.train_on_minibatch.assert_not_called()

    def test_update_value_function_reset_target_network(self):
        with patch('random.sample', side_effect=lambda lst, n: lst[-n:]):
            self.algo.run_gpi_for_an_episode(self.task, self.policy, self.value_func)
        self.eq(2, self.algo.reset_step_counter)
        self.eq("Q_hat_network_0", self.value_func.q_hat_network.name)
        self.algo.run_gpi_for_an_episode(self.task, self.policy, self.value_func)
        self.eq(0, self.algo.reset_step_counter)
        self.eq("Q_hat_network_1", self.value_func.q_hat_network.name)

    def test_save_and_load_algorithm_state(self):
        dir_path = generate_tmp_dir_path(__file__)
        file_path = os.path.join(dir_path, "dq_replay_memory.pickle")
        setup_tmp_dir(__file__)
        self.algo.save_algorithm_state(dir_path)
        self.true(os.path.exists(file_path))

        new_algo = DeepQLearning(replay_start_size=100)
        task = setup_stub_task()
        # Overrider terminal judge logic to avoid infinite episode by random policy
        task.is_terminal_state.side_effect = lambda state: state == 4 or state >= 100
        policy = NegativePolicy()
        value_func = self.DeepQLearningApproxActionValueFunctionImpl(strict_mode=False)
        new_algo.setup(task, policy, value_func)
        new_algo.load_algorithm_state(dir_path)

        # Validate algorithm's state
        self.eq(self.algo.gamma, new_algo.gamma)
        self.eq(self.algo.C, new_algo.C)
        self.eq(self.algo.minibatch_size, new_algo.minibatch_size)
        self.eq(self.algo.replay_start_size, new_algo.replay_start_size)
        self.eq(self.algo.reset_step_counter, new_algo.reset_step_counter)
        self.eq(task, new_algo.task)
        self.eq(policy, new_algo.policy)
        self.eq(value_func, new_algo.value_function)
        self.eq(self.algo.replay_memory.max_size, new_algo.replay_memory.max_size)
        self.eq(self.algo.replay_memory.queue, new_algo.replay_memory.queue)
        self.true(isinstance(new_algo.greedy_policy, GreedyPolicy))

       # Validate that loaded algorithm works like original one
        with patch('random.sample', side_effect=lambda lst, n: lst[len(lst)-n:]):
            new_algo.run_gpi_for_an_episode(self.task, self.policy, self.value_func)
        replay_memory_expected = [
                (5.0, 7, 144, 4),
                (0, 1, 1, 1),
                (1, 3, 16, 4)
        ]
        self.eq(replay_memory_expected, new_algo.replay_memory.queue)

    class DeepQLearningApproxActionValueFunctionImpl(DeepQLearningApproxActionValueFunction):

        def __init__(self, strict_mode=True):
            self.deepcopy_counter = 0
            self.strict_mode = strict_mode

        def initialize_network(self):
            mock_q_network = Mock(name="Q_network")
            mock_q_network.predict.side_effect = self.q_predict_scenario
            return mock_q_network

        def deepcopy_network(self, q_network):
            mock_q_hat_network = Mock(name="Q_hat_network")
            mock_q_hat_network.name = "Q_hat_network_%d" % self.deepcopy_counter
            mock_q_hat_network.predict.side_effect = self.q_hat_predict_scenario
            self.deepcopy_counter += 1
            return mock_q_hat_network

        def predict_value_by_network(self, network, state, action):
            return network.predict(state, action)

        def backup_on_minibatch(self, q_network, backup_minibatch):
            q_network.train_on_minibatch(backup_minibatch)

        def q_predict_scenario(self, state, action):
            if state == 0 and action == 1:
                return 1
            elif state == 0 and action == 2:
                return 2
            elif state == 1 and action == 2:
                return 4
            elif state == 1 and action == 3:
                return 3
            else:
                if self.strict_mode:
                    raise AssertionError("q_network received unexpected state-action pair (state=%s, action=%s)" % (state, action))
                else:
                    return 1

        def q_hat_predict_scenario(self, state, action):
            if state == 1 and action == 2:
                return 5
            elif state == 1 and action == 3:
                return 6
            else:
                if self.strict_mode:
                    raise AssertionError("q_hat_network received unexpected state-action pair (state=%s, action=%s)" % (state, action))
                else:
                    return 1

class DeepQLearningApproxValueFunctionTest(BaseUnitTest):

    def setUp(self):
        self.empty_func = DeepQLearningApproxActionValueFunction()
        self.func = self.DeepQLearningApproxActionValueFunctionImpl()
        self.func.setup()

    def tearDown(self):
        cleanup_trash()

    @raises(NotImplementedError)
    def test_initialize_network(self):
        self.empty_func.initialize_network()

    @raises(NotImplementedError)
    def test_deepcopy_network(self):
        self.empty_func.deepcopy_network("dummy")

    @raises(NotImplementedError)
    def test_predict_value_by_network(self):
        self.empty_func.predict_value_by_network("dummy", "dummy", "dummy")

    @raises(NotImplementedError)
    def test_backup_on_minibatch(self):
        self.empty_func.backup_on_minibatch("dummy", "dummy")

    @raises(NotImplementedError)
    def test_save_networks(self):
        self.empty_func.save_networks("dummy", "dummy", "dummy")

    @raises(NotImplementedError)
    def test_load_networks(self):
        self.empty_func.load_networks("dummy")

    def test_setup(self):
        self.eq(0, self.func.q_network)
        self.eq(1, self.func.q_hat_network)
        self.false(self.func.use_target_network_flg)

    def test_use_target_network(self):
        self.func.use_target_network(True)
        self.true(self.func.use_target_network_flg)
        self.func.use_target_network(False)
        self.false(self.func.use_target_network_flg)

    def test_calculate_value(self):
        self.eq(0, self.func.predict_value(2, 3))
        self.func.use_target_network(True)
        self.eq(6, self.func.predict_value(2, 3))

    def test_reset_target_network(self):
        self.func.q_network = 2
        self.func.reset_target_network()
        self.eq(3, self.func.q_hat_network)

    def test_save_and_load_networks(self):
        self.func.q_network = 2
        dir_path = generate_tmp_dir_path(__file__)
        setup_tmp_dir(__file__)
        self.func.save(dir_path)
        new_func = self.DeepQLearningApproxActionValueFunctionImpl()
        new_func.setup()
        new_func.load(dir_path)
        self.eq(self.func.q_network, new_func.q_network)
        self.eq(self.func.q_hat_network, new_func.q_hat_network)

    class DeepQLearningApproxActionValueFunctionImpl(DeepQLearningApproxActionValueFunction):

        def initialize_network(self):
            return 0

        def deepcopy_network(self, q_network):
            return q_network + 1

        def predict_value_by_network(self, network, state, action):
            return network * state * action

        def save_networks(self, q_network, q_hat_network, save_dir_path):
            pickle_data(gen_tmp_file_path(save_dir_path), (q_network, q_hat_network))

        def load_networks(self, load_dir_path):
            q_network, q_hat_network = unpickle_data(gen_tmp_file_path(load_dir_path))
            return q_network, q_hat_network


class ExperienceReplayTest(BaseUnitTest):

    def test_queue_size_upper_bound(self):
        er = ExperienceReplay(max_size=2)
        experiences = [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9 ,0, 1)]
        for e in experiences:
            er.store_transition(state=e[0], action=e[1], reward=e[2], next_state=e[3])

        self.neq(er.queue[0], experiences[0])
        self.eq(er.queue[0], experiences[1])
        self.eq(er.queue[1], experiences[2])

    def test_sample_minibatch(self):
        er = ExperienceReplay(max_size=3)
        experiences = [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9 ,0, 1)]
        for e in experiences:
            er.store_transition(state=e[0], action=e[1], reward=e[2], next_state=e[3])
        self.eq(1, len(er.sample_minibatch(minibatch_size=1)))
        with patch('random.sample', side_effect=lambda lst, num: lst[len(lst)-num:]):
            minibatch = er.sample_minibatch(minibatch_size=2)
            expected = experiences[1:]
            self.eq(expected, minibatch)

    def test_dump_load(self):
        er = ExperienceReplay(max_size=2)
        experiences = [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9 ,0, 1)]
        for e in experiences:
            er.store_transition(state=e[0], action=e[1], reward=e[2], next_state=e[3])
        dump = er.dump()

        new_er = ExperienceReplay(max_size=3)
        new_er.load(dump)
        self.eq(er.max_size, new_er.max_size)
        self.eq(er.queue, new_er.queue)


def setup_stub_task():
    mock_task = Mock()
    mock_task.generate_initial_state.return_value = 0
    mock_task.is_terminal_state.side_effect = lambda state: state == 4
    mock_task.transit_state.side_effect = lambda state, action: state + action
    mock_task.generate_possible_actions.side_effect = lambda state: [] if state == 4 else [state + 1, state + 2]
    mock_task.calculate_reward.side_effect = lambda state: state**2
    return mock_task

def gen_tmp_file_path(dir_path):
    return os.path.join(dir_path, "hoge.pickle")

def cleanup_trash():
    filenames = ["hoge.pickle", "dq_replay_memory.pickle"]
    teardown_tmp_dir(__file__, filenames)


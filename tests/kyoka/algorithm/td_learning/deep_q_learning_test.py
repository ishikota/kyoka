from tests.base_unittest import BaseUnitTest
from kyoka.algorithm.td_learning.deep_q_learning import DeepQLearning
from kyoka.value_function.base_deep_q_learning_action_value_function import BaseDeepQLearningActionValueFunction
from kyoka.policy.base_policy import BasePolicy
from kyoka.policy.greedy_policy import GreedyPolicy

from mock import Mock
from mock import patch

import os

class DeepQLearningTest(BaseUnitTest):

  def setUp(self):
    self.algo = DeepQLearning(gamma=0.1, N=3, C=3, minibatch_size=2, replay_start_size=2)
    self.algo.replay_memory.store_transition(2.5, 3, 25, (5, False))
    self.algo.replay_memory.store_transition(5.0, 7, 144, (5.5, True))
    self.domain = self.__setup_stub_domain()
    self.value_func = self.TestValueFunctionImpl()
    self.policy = self.NegativePolicyImple()
    self.algo.setUp(self.domain, self.policy, self.value_func)

  def tearDown(self):
    dir_path = self.__generate_tmp_dir_path()
    file_path = os.path.join(dir_path, "dqn_algorithm_state.pickle")
    if os.path.exists(dir_path):
      if os.path.exists(file_path):
        os.remove(file_path)
      os.rmdir(dir_path)

  def test_update_value_function_learning_minibatch_delivery(self):
    with patch('random.sample', side_effect=lambda lst, n: lst[len(lst)-n:]):
      self.algo.update_value_function(self.domain, self.policy, self.value_func)

    learning_minibatch_expected = [
        [(5.0, 7, 144), (0.5, 1.5, 2.85)],
        [(0.5, 1.5, 2.85), (2, 4, 30.25)]
    ]
    actual = [arg[0][0] for arg in self.value_func.Q.train_on_minibatch.call_args_list]
    self.eq(learning_minibatch_expected, actual)
    self.value_func.Q_hat.train_on_minibatch.assert_not_called()

  def test_update_value_function_experience_replay_memory_state(self):
    with patch('random.sample', side_effect=lambda lst, n: lst[len(lst)-n:]):
      self.algo.update_value_function(self.domain, self.policy, self.value_func)
    replay_memory_expected = [
        (5.0, 7, 144, (5.5, True)),
        (0.5, 1.5, 2.25, (2, False)),
        (2, 4, 30.25, (6.0, True))
    ]
    self.eq(replay_memory_expected, self.algo.replay_memory.queue)

  def test_update_value_function_reset_target_network(self):
    with patch('random.sample', side_effect=lambda lst, n: lst[-n:]):
      self.algo.update_value_function(self.domain, self.policy, self.value_func)
    self.eq(2, self.algo.reset_step_counter)
    self.eq("Q_hat_network_0", self.value_func.Q_hat.name)
    self.algo.update_value_function(self.domain, self.policy, self.value_func)
    self.eq(0, self.algo.reset_step_counter)
    self.eq("Q_hat_network_1", self.value_func.Q_hat.name)

  def test_initialize_replay_memory(self):
    algo = DeepQLearning(gamma=0.1, N=3, C=3, minibatch_size=2, replay_start_size=2)
    value_func = self.TestValueFunctionImpl(strict_mode=False)
    # Overrider terminal judge logic to avoid infinite episode by random policy
    self.domain.is_terminal_state.side_effect = lambda state: state == 4 or state >= 100
    self.eq(0, len(algo.replay_memory.queue))
    algo.setUp(self.domain, self.policy, value_func)
    self.eq(2, len(algo.replay_memory.queue))

  def test_save_and_load_algorithm_state(self):
    dir_path = self.__generate_tmp_dir_path()
    file_path = os.path.join(dir_path, "dqn_algorithm_state.pickle")
    os.mkdir(dir_path)

    self.algo.save_algorithm_state(dir_path)
    self.true(os.path.exists(file_path))

    new_algo = DeepQLearning(replay_start_size=100)
    domain = self.__setup_stub_domain()
    # Overrider terminal judge logic to avoid infinite episode by random policy
    domain.is_terminal_state.side_effect = lambda state: state == 4 or state >= 100
    value_func = self.TestValueFunctionImpl(strict_mode=False)
    policy = self.NegativePolicyImple()
    new_algo.setUp(domain, policy, value_func)
    new_algo.load_algorithm_state(dir_path)

    # Validate algorithm's state
    self.eq(self.algo.gamma, new_algo.gamma)
    self.eq(self.algo.C, new_algo.C)
    self.eq(self.algo.minibatch_size, new_algo.minibatch_size)
    self.eq(self.algo.replay_start_size, new_algo.replay_start_size)
    self.eq(self.algo.reset_step_counter, new_algo.reset_step_counter)
    self.eq(domain, new_algo.domain)
    self.eq(policy, new_algo.policy)
    self.eq(value_func, new_algo.value_function)
    self.eq(self.algo.replay_memory.max_size, new_algo.replay_memory.max_size)
    self.eq(self.algo.replay_memory.queue, new_algo.replay_memory.queue)
    self.true(isinstance(new_algo.greedy_policy, GreedyPolicy))

    # Validate that loaded algorithm works like original one
    with patch('random.sample', side_effect=lambda lst, n: lst[len(lst)-n:]):
      new_algo.update_value_function(self.domain, self.policy, self.value_func)
    replay_memory_expected = [
        (5.0, 7, 144, (5.5, True)),
        (0.5, 1.5, 2.25, (2, False)),
        (2, 4, 30.25, (6.0, True))
    ]
    self.eq(replay_memory_expected, new_algo.replay_memory.queue)

  def __setup_stub_domain(self):
    mock_domain = Mock()
    mock_domain.generate_initial_state.return_value = 0
    mock_domain.is_terminal_state.side_effect = lambda state: state == 5.5
    mock_domain.transit_state.side_effect = lambda state, action: state + action
    mock_domain.generate_possible_actions.side_effect = lambda state: [] if state == 5.5 else [state + 1, state + 2]
    mock_domain.calculate_reward.side_effect = lambda state: state**2
    return mock_domain

  def __generate_tmp_dir_path(self):
    return os.path.join(os.path.dirname(__file__), "tmp")

  class TestValueFunctionImpl(BaseDeepQLearningActionValueFunction):

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

    def preprocess_state_sequence(self, raw_state_sequence):
      return raw_state_sequence[-1] + 0.5

    def predict_action_value(self, network, processed_state, action):
      return network.predict(processed_state, action)

    def train_on_minibatch(self, network, learning_minibatch):
      network.train_on_minibatch(learning_minibatch)

    def q_predict_scenario(self, state, action):
      if state == 0.5 and action == 1.5:
        return 1
      elif state == 0.5 and action == 2.5:
        return 2
      elif state == 2 and action == 3:
        return 4
      elif state == 2 and action == 4:
        return 3
      else:
        if self.strict_mode:
          raise AssertionError("q_network received unexpected state-action pair (state=%s, action=%s)" % (state, action))
        else:
          return 1

    def q_hat_predict_scenario(self, state, action):
      if state == 2 and action == 3:
        return 5
      elif state == 2 and action == 4:
        return 6
      else:
        if self.strict_mode:
          raise AssertionError("q_hat_network received unexpected state-action pair (state=%s, action=%s)" % (state, action))
        else:
          return 1

  class NegativePolicyImple(BasePolicy):

    def choose_action(self, domain, value_function, state):
      actions = domain.generate_possible_actions(state)
      calc_Q_value = lambda state, action: value_function.calculate_value(state, action)
      Q_value_for_actions = [calc_Q_value(state, action) for action in actions]
      min_Q_value = min(Q_value_for_actions)
      Q_act_pair = zip(Q_value_for_actions, actions)
      worst_actions = [act for Q_value, act in Q_act_pair if min_Q_value == Q_value]
      return worst_actions[0]


from tests.base_unittest import BaseUnitTest
from kyoka.algorithm.td_learning.deep_q_learning import DeepQLearning
from kyoka.value_function.base_deep_q_learning_action_value_function import BaseDeepQLearningActionValueFunction
from kyoka.policy.base_policy import BasePolicy

from mock import Mock
from mock import patch

class DeepQLearningTest(BaseUnitTest):

  def setUp(self):
    self.algo = DeepQLearning(gamma=0.1, N=3, C=3, minibatch_size=2, replay_start_size=2)
    self.algo.replay_memory.store_transition(2.5, 3, 25, (5, False))
    self.algo.replay_memory.store_transition(5.0, 7, 144, (4, True))
    self.domain = self.__setup_stub_domain()
    self.value_func = self.TestValueFunctionImpl()
    self.value_func.setUp()
    self.policy = self.NegativePolicyImple(self.domain, self.value_func)

  def test_update_value_function_learning_minibatch_delivery(self):
    with patch('random.sample', side_effect=lambda lst, n: lst[len(lst)-n:]):
      self.algo.update_value_function(self.domain, self.policy, self.value_func)

    learning_minibatch_expected = [
        [(5.0, 7, 144), (0.5, 1, 1.6)],
        [(0.5, 1, 1.6), (1.5, 3, 16)]
    ]
    actual = [arg[0][0] for arg in self.value_func.Q.train_on_minibatch.call_args_list]
    self.eq(learning_minibatch_expected, actual)
    self.value_func.Q_hat.train_on_minibatch.assert_not_called()

  def test_update_value_function_experience_replay_memory_state(self):
    with patch('random.sample', side_effect=lambda lst, n: lst[len(lst)-n:]):
      self.algo.update_value_function(self.domain, self.policy, self.value_func)
    replay_memory_expected = [
        (5.0, 7, 144, (4, True)),
        (0.5, 1, 1, (1.5, False)),
        (1.5, 3, 16, (4.5, True))
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
    value_func.setUp()
    # Overrider terminal judge logic to avoid infinite episode by random policy
    self.domain.is_terminal_state.side_effect = lambda state: state == 4 or state >= 100
    self.eq(0, len(algo.replay_memory.queue))
    algo.update_value_function(self.domain, self.policy, value_func)
    self.eq(3, len(algo.replay_memory.queue))


  def __setup_stub_domain(self):
    mock_domain = Mock()
    mock_domain.generate_initial_state.return_value = 0
    mock_domain.is_terminal_state.side_effect = lambda state: state == 4
    mock_domain.transit_state.side_effect = lambda state, action: state + action
    mock_domain.generate_possible_actions.side_effect = lambda state: [] if state == 4 else [state + 1, state + 2]
    mock_domain.calculate_reward.side_effect = lambda state: state**2
    return mock_domain

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

    def preprocess_state(self, state):
      return state + 0.5

    def predict_action_value(self, network, processed_state, action):
      return network.predict(processed_state, action)

    def train_on_minibatch(self, network, learning_minibatch):
      network.train_on_minibatch(learning_minibatch)

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
      if state == 1.5 and action == 2.5:
        return 5
      elif state == 1.5 and action == 3.5:
        return 6
      elif state == 4.5 and action == 5.5:
        return 8
      elif state == 4.5 and action == 6.5:
        return 7
      else:
        if self.strict_mode:
          raise AssertionError("q_hat_network received unexpected state-action pair (state=%s, action=%s)" % (state, action))
        else:
          return 1

  class NegativePolicyImple(BasePolicy):

    def choose_action(self, state):
      actions = self.domain.generate_possible_actions(state)
      calc_Q_value = lambda state, action: self.value_function.calculate_value(state, action)
      Q_value_for_actions = [calc_Q_value(state, action) for action in actions]
      min_Q_value = min(Q_value_for_actions)
      Q_act_pair = zip(Q_value_for_actions, actions)
      worst_actions = [act for Q_value, act in Q_act_pair if min_Q_value == Q_value]
      return worst_actions[0]


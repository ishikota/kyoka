from kyoka.algorithm.td_learning.base_td_method import BaseTDMethod
from kyoka.algorithm.experience_replay.experience_replay import ExperienceReplay
from kyoka.policy.greedy_policy import GreedyPolicy
from kyoka.policy.epsilon_greedy_policy import EpsilonGreedyPolicy

import os
import pickle

class DeepQLearning(BaseTDMethod):

  SAVE_FILE_NAME = "dqn_algorithm_state.pickle"
  ACTION_ON_TERMINAL_FLG = "action_on_terminal"

  def __init__(self, gamma=0.99, N=1000000, C=10000, minibatch_size=32, replay_start_size=50000):
    BaseTDMethod.__init__(self)
    self.gamma = gamma
    self.replay_memory = ExperienceReplay(max_size=N)
    self.C = C
    self.minibatch_size = minibatch_size
    self.replay_start_size = replay_start_size
    self.reset_step_counter = 0

  def setUp(self, domain, policy, value_function):
    super(DeepQLearning, self).setUp(domain, policy, value_function)
    self.__initialize_replay_memory(domain, value_function, self.replay_memory, self.replay_start_size)
    self.greedy_policy = GreedyPolicy()

  def save_algorithm_state(self, save_dir_path):
    state = (
            self.gamma, self.replay_memory.dump(), self.C, self.minibatch_size,
            self.replay_start_size, self.reset_step_counter
    )
    self.__pickle_data(self.__gen_save_file_path(save_dir_path), state)

  def load_algorithm_state(self, load_dir_path):
    state = self.__unpickle_data(self.__gen_save_file_path(load_dir_path))
    self.gamma, replay_memory_serial, self.C, self.minibatch_size,\
            self.replay_start_size, self.reset_step_counter = state
    self.greedy_policy = GreedyPolicy()
    new_replay_memory = ExperienceReplay(max_size=1)
    new_replay_memory.load(replay_memory_serial)
    self.replay_memory = new_replay_memory

  def update_action_value_function(self, domain, policy, value_function):
    phi = lambda s: value_function.preprocess_state_sequence(s)
    value_function.use_target_network(False)
    state = domain.generate_initial_state()
    sequence = [state]

    while not domain.is_terminal_state(sequence[-1]):
      action = policy.choose_action(domain, value_function, phi(sequence))
      next_state = domain.transit_state(sequence[-1], action)
      reward = domain.calculate_reward(next_state)
      sequence.append(next_state)
      next_state_info = (phi(sequence), domain.is_terminal_state(next_state))

      self.replay_memory.store_transition(phi(sequence[:-1]), action, reward, next_state_info)
      experience_minibatch = self.replay_memory.sample_minibatch(self.minibatch_size)
      learning_minibatch= self.__gen_learning_minibatch(self.domain, self.greedy_policy, value_function, experience_minibatch)
      value_function.train_on_minibatch(value_function.Q, learning_minibatch)

      if self.reset_step_counter >= self.C:
        value_function.reset_target_network()
        self.reset_step_counter = 0
      else:
        self.reset_step_counter += 1


  def __initialize_replay_memory(self, domain, value_function, replay_memory, start_size):
    phi = lambda s: value_function.preprocess_state_sequence(s)
    random_policy = EpsilonGreedyPolicy(eps=1.0)
    while len(replay_memory.queue) < start_size:
      sequence = [domain.generate_initial_state()]
      for _, action, next_state, reward in self.generate_episode(domain, value_function, random_policy):
        sequence.append(next_state)
        next_state_info = (phi(sequence), domain.is_terminal_state(next_state))
        replay_memory.store_transition(phi(sequence[:-1]), action, reward, next_state_info)
        if len(replay_memory.queue) >= start_size: return


  def __gen_learning_minibatch(self, domain, greedy_policy, value_function, experiences):
    value_function.use_target_network(True)
    targets = [self.__gen_learning_data(domain, greedy_policy, value_function, experience) for experience in experiences]
    value_function.use_target_network(False)
    return targets

  def __gen_learning_data(self, domain, greedy_policy, value_function, experience):
    state, action, reward, next_state_info = experience
    next_state, is_terminal = next_state_info
    greedy_action = self.__choose_action(greedy_policy, domain, value_function, next_state, is_terminal)
    greedy_Q_value = self.__calculate_value(value_function, next_state, greedy_action)
    target = reward + self.gamma * greedy_Q_value
    return (state, action, target)

  def __choose_action(self, policy, domain, value_function, state, is_terminal_state):
    if is_terminal_state:
      return self.ACTION_ON_TERMINAL_FLG
    else:
      return policy.choose_action(domain, value_function, state)

  def __calculate_value(self, value_function, next_state, next_action):
    if self.ACTION_ON_TERMINAL_FLG == next_action:
      return 0
    else:
      return value_function.calculate_value(next_state, next_action)

  def __gen_save_file_path(self, base_dir_path):
    return os.path.join(base_dir_path, self.SAVE_FILE_NAME)

  def __pickle_data(self, file_path, data):
    with open(file_path, "wb") as f:
      pickle.dump(data, f)

  def __unpickle_data(self, file_path):
    with open(file_path, "rb") as f:
      return pickle.load(f)


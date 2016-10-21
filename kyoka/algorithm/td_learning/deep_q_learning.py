from kyoka.algorithm.td_learning.base_td_method import BaseTDMethod
from kyoka.algorithm.experience_replay.experience_replay import ExperienceReplay
from kyoka.policy.greedy_policy import GreedyPolicy
from kyoka.policy.epsilon_greedy_policy import EpsilonGreedyPolicy

class DeepQLearning(BaseTDMethod):

  ACTION_ON_TERMINAL_FLG = "action_on_terminal"

  def __init__(self, gamma=0.99, N=1000000, C=10000, minibatch_size=32, replay_start_size=50000):
    BaseTDMethod.__init__(self)
    self.gamma = gamma
    self.replay_memory = ExperienceReplay(max_size=N)
    self.C = C
    self.minibatch_size = minibatch_size
    self.replay_start_size = replay_start_size
    self.sync_step_counter = 0

  def update_action_value_function(self, domain, policy, value_function):
    phi = lambda s: value_function.preprocess_state(s)
    self.__initialize_replay_memory_if_needed(domain, value_function, self.replay_memory, self.replay_start_size)
    value_function.use_target_network(False)
    greedy_policy = GreedyPolicy(domain, value_function)
    state = domain.generate_initial_state()

    while not domain.is_terminal_state(state):
      action = policy.choose_action(state)
      next_state = domain.transit_state(state, action)
      reward = domain.calculate_reward(next_state)
      next_state_info = (phi(next_state), domain.is_terminal_state(next_state))

      self.replay_memory.store_transition(phi(state), action, reward, next_state_info)
      experience_minibatch = self.replay_memory.sample_minibatch(self.minibatch_size)
      learning_minibatch= self.__gen_learning_minibatch(greedy_policy, value_function, experience_minibatch)
      value_function.train_on_minibatch(value_function.Q, learning_minibatch)

      if self.sync_step_counter >= self.C:
        value_function.sync_target_network()
        self.sync_step_counter = 0
      else:
        self.sync_step_counter += 1

      state = next_state


  def __initialize_replay_memory_if_needed(self, domain, value_function, replay_memory, start_size):
    if len(replay_memory.queue) < start_size:
      self.__initialize_replay_memory(domain, value_function, replay_memory, start_size)

  def __initialize_replay_memory(self, domain, value_function, replay_memory, start_size):
    phi = lambda s: value_function.preprocess_state(s)
    random_policy = EpsilonGreedyPolicy(domain, value_function, eps=1.0)
    while len(replay_memory.queue) < start_size:
      for state, action, next_state, reward in self.generate_episode(domain, random_policy):
        next_state_info = (phi(next_state), domain.is_terminal_state(next_state))
        replay_memory.store_transition(phi(state), action, reward, next_state_info)

  def __gen_learning_minibatch(self, greedy_policy, value_function, experiences):
    value_function.use_target_network(True)
    targets = [self.__gen_learning_data(greedy_policy, value_function, experience) for experience in experiences]
    value_function.use_target_network(False)
    return targets

  def __gen_learning_data(self, greedy_policy, value_function, experience):
    state, action, reward, next_state_info = experience
    next_state, is_terminal = next_state_info
    greedy_action = self.__choose_action(greedy_policy, next_state, is_terminal)
    greedy_Q_value = self.__calculate_value(value_function, next_state, greedy_action)
    target = reward + self.gamma * greedy_Q_value
    return (state, action, target)

  def __choose_action(self, policy, state, is_terminal_state):
    if is_terminal_state:
      return self.ACTION_ON_TERMINAL_FLG
    else:
      return policy.choose_action(state)

  def __calculate_value(self, value_function, next_state, next_action):
    if self.ACTION_ON_TERMINAL_FLG == next_action:
      return 0
    else:
      return value_function.calculate_value(next_state, next_action)


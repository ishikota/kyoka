import random
from kyoka.algorithm.policy.base_policy import BasePolicy

class EpsilonGreedyPolicy(BasePolicy):

  def __init__(self, domain, value_function, eps=0.05, rand=None):
    super(EpsilonGreedyPolicy, self).__init__(domain, value_function)
    self.eps = eps
    self.rand = rand if rand else random

  def choose_action(self, Q, state):
    actions = self.domain.generate_possible_actions(state)
    best_action = self.__choose_best_action(Q, state)
    probs = self.__calc_select_probability(best_action, actions)
    selected_action_idx = self.__roulette(probs)
    return actions[selected_action_idx]

  def __choose_best_action(self, Q, state):
    actions = self.domain.generate_possible_actions(state)
    pack = lambda state, action: self.pack_arguments_for_value_function(state, action)
    calc_Q_value = lambda packed_arg: self.value_function.calculate_value(*packed_arg)
    Q_value_for_actions = [calc_Q_value(pack(state, action)) for action in actions]
    max_Q_value = max(Q_value_for_actions)
    Q_act_pair = zip(Q_value_for_actions, actions)
    best_actions = [act for Q_value, act in Q_act_pair if max_Q_value == Q_value]
    best_action = self.rand.choice(best_actions)
    return best_action

  def __calc_select_probability(self, best_action, actions):
    e = self.eps / len(actions)
    bonus = 1 - self.eps
    calc_prob = lambda action: e + bonus if action == best_action else e
    return [calc_prob(action) for action in actions]

  def __roulette(self, probs):
    hit_idx = -1
    prob_sum = 0
    dart = self.rand.random()
    for idx, prob in enumerate(probs):
      prob_sum += prob
      if dart < prob_sum:
        hit_idx = idx
        break
    return hit_idx


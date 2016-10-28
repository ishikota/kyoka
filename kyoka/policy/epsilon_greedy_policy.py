import random
from kyoka.policy.base_policy import BasePolicy

class EpsilonGreedyPolicy(BasePolicy):

  def __init__(self, eps=0.05, rand=None):
    self.eps = eps
    self.rand = rand if rand else random
    self.do_annealing = False

  def choose_action(self, domain, value_function, state):
    actions = domain.generate_possible_actions(state)
    best_action = self.__choose_best_action(domain, value_function, state)
    probs = self.__calc_select_probability(best_action, actions)
    selected_action_idx = self.__roulette(probs)
    return actions[selected_action_idx]

  def set_eps_annealing(self, initial_eps, final_eps, anneal_duration):
    self.do_annealing = True
    self.eps = initial_eps
    self.min_eps = final_eps
    self.anneal_step = (initial_eps - final_eps) / anneal_duration

  def anneal_eps(self):
    self.eps = max(self.min_eps, self.eps - self.anneal_step)

  def __choose_best_action(self, domain, value_func, state):
    actions = domain.generate_possible_actions(state)
    pack = lambda state, action: self.pack_arguments_for_value_function(value_func, state, action)
    calc_Q_value = lambda packed_arg: value_func.calculate_value(*packed_arg)
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


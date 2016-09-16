import random
from kyoka.algorithm.policy.base_policy import BasePolicy

class GreedyPolicy(BasePolicy):

  def __init__(self, domain, rand=None):
    super(GreedyPolicy, self).__init__(domain)
    self.rand = rand if rand else random

  def choose_action(self, Q, state):
    return self.__choose_best_action(Q, state)

  def __choose_best_action(self, Q, state):
    actions = self.domain.generate_possible_actions(state)
    Q_value_for_actions = [self.domain.fetch_Q_value(Q, state, action) for action in actions]
    max_Q_value = max(Q_value_for_actions)
    Q_act_pair = zip(Q_value_for_actions, actions)
    best_actions = [act for Q_value, act in Q_act_pair if max_Q_value == Q_value]
    best_action = self.rand.choice(best_actions)
    return best_action


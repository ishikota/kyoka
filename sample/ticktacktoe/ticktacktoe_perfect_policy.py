from kyoka.policy.base_policy import BasePolicy
import random
import logging

class TickTackToePerfectPolicy(BasePolicy):

  def choose_action(self, state):
    actions = self.domain.generate_possible_actions(state)
    states = [self.domain.transit_state(state, action) for action in actions]
    values = [self.mini(state, -20, 20) for state in states]
    logging.debug("MiniMax calculation result [(action, score),...] => %s" % zip(actions, values))
    best_actions = [act for act, val in zip(actions, values) if val == max(values)]
    return random.choice(best_actions)


  def maxi(self, state, alpha, beta):
    if self.domain.is_terminal_state(state):
      return self.domain.calculate_reward(state)
    for action in self.domain.generate_possible_actions(state):
      next_state = self.domain.transit_state(state, action)
      score = self.mini(next_state, alpha, beta)
      if score >= beta:
        return beta
      if score > alpha:
        alpha = score
    return alpha

  def mini(self, state, alpha, beta):
    if self.domain.is_terminal_state(state):
      return self.domain.calculate_reward(state)
    best = 100
    for action in self.domain.generate_possible_actions(state):
      next_state = self.domain.transit_state(state, action)
      score = self.maxi(next_state, alpha, beta)
      if score <= alpha:
        return alpha
      if score < beta:
        beta = score
    return beta


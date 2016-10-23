from kyoka.policy.base_policy import BasePolicy
import random
import logging

class TickTackToePerfectPolicy(BasePolicy):

  def choose_action(self, domain, value_function, state):
    actions = domain.generate_possible_actions(state)
    states = [domain.transit_state(state, action) for action in actions]
    values = [self.mini(domain, state, -20, 20) for state in states]
    logging.debug("MiniMax calculation result [(action, score),...] => %s" % zip(actions, values))
    best_actions = [act for act, val in zip(actions, values) if val == max(values)]
    return random.choice(best_actions)


  def maxi(self, domain, state, alpha, beta):
    if domain.is_terminal_state(state):
      return domain.calculate_reward(state)
    for action in domain.generate_possible_actions(state):
      next_state = domain.transit_state(state, action)
      score = self.mini(domain, next_state, alpha, beta)
      if score >= beta:
        return beta
      if score > alpha:
        alpha = score
    return alpha

  def mini(self, domain, state, alpha, beta):
    if domain.is_terminal_state(state):
      return domain.calculate_reward(state)
    best = 100
    for action in domain.generate_possible_actions(state):
      next_state = domain.transit_state(state, action)
      score = self.maxi(domain, next_state, alpha, beta)
      if score <= alpha:
        return alpha
      if score < beta:
        beta = score
    return beta


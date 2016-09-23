from kyoka.policy.base_policy import BasePolicy
import random

class TickTackToePerfectPolicy(BasePolicy):

  def choose_action(self, state):
    actions = self.domain.generate_possible_actions(state)
    states = [self.domain.transit_state(state, action) for action in actions]
    values = [self.mini(state) for state in states]
    print zip(actions, values)
    best_actions = [act for act, val in zip(actions, values) if val == max(values)]
    return random.choice(best_actions)


  def maxi(self, state):
    if self.domain.is_terminal_state(state):
      return self.domain.calculate_reward(state)
    best = -100
    for action in self.domain.generate_possible_actions(state):
      next_state = self.domain.transit_state(state, action)
      score = self.mini(next_state)
      if score > best:
        best = score
    return best

  def mini(self, state):
    if self.domain.is_terminal_state(state):
      return self.domain.calculate_reward(state)
    best = 100
    for action in self.domain.generate_possible_actions(state):
      next_state = self.domain.transit_state(state, action)
      score = self.maxi(next_state)
      if score < best:
        best = score
    return best


from kyoka.policy.base_policy import BasePolicy

class TickTackToeManualPolicy(BasePolicy):

  ACTION_NAME_MAP = {
        1 : "lower_right",
        2 : "lower_center",
        4 : "lower_left",
        8 : "middle_right",
        16: "middle_center",
        32: "middle_left",
        64: "upper_right",
        128: "upper_center",
        256: "upper_left"
  }

  def choose_action(self, state):
    message = self.__ask_message(state) + " >> "
    action = int(raw_input(message))
    if action not in self.domain.generate_possible_actions(state):
      return self.choose_action(state)
    return action

  def __ask_message(self, state):
    possible_actions = self.domain.generate_possible_actions(state)
    names = [self.ACTION_NAME_MAP[action] for action in possible_actions]
    return ", ".join(["%d: %s" % info for info in zip(possible_actions, names)])


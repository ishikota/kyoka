from utils import build_not_implemented_msg


class BaseTask(object):

  def generate_inital_state(self):
    err_msg = build_not_implemented_msg(self, "generate_inital_state")
    raise NotImplementedError(err_msg)

  def is_terminal_state(self, state):
    err_msg = build_not_implemented_msg(self, "is_terminal_state")
    raise NotImplementedError(err_msg)

  def transit_state(self, state, action):
    err_msg = build_not_implemented_msg(self, "transit_state")
    raise NotImplementedError(err_msg)

  def generate_possible_actions(self, state):
    err_msg = build_not_implemented_msg(self, "generate_possible_actions")
    raise NotImplementedError(err_msg)

  def calculate_reward(self, state):
    err_msg = build_not_implemented_msg(self, "calculate_reward")
    raise NotImplementedError(err_msg)


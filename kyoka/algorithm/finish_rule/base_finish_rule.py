class BaseFinishRule(object):

  def satisfy_condition(self, iteration_count, deltas):
    err_msg = self.__build_err_msg("satisfy_condition")
    raise NotImplementedError(err_msg)

  def generate_finish_message(self, iteration_count, deltas):
    err_msg = self.__build_err_msg("generate_finish_message")
    raise NotImplementedError(err_msg)

  def __build_err_msg(self, msg):
    base_msg = "[ {0} ] class does not implement [ {1} ] method"
    return base_msg.format(self.__class__.__name__, msg)


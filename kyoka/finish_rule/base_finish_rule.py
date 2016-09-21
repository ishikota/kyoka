import logging

class BaseFinishRule(object):

  def __init__(self, log_interval=100):
    self.log_interval = log_interval
    self.log_interval_counter = 0

  def satisfy_condition(self, iteration_count, deltas):
    is_satisfied = self.check_condition(iteration_count, deltas)
    self.__notify_message_if_needed(is_satisfied, iteration_count, deltas)
    return is_satisfied

  def check_condition(self, iteration_count, deltas):
    err_msg = self.__build_err_msg("check_condition")
    raise NotImplementedError(err_msg)

  def generate_progress_message(self, iteration_count, deltas):
    err_msg = self.__build_err_msg("generate_progress_message")
    raise NotImplementedError(err_msg)

  def generate_finish_message(self, iteration_count, deltas):
    err_msg = self.__build_err_msg("generate_finish_message")
    raise NotImplementedError(err_msg)

  def __notify_message_if_needed(self, is_satisfied_condition, iteration_count, deltas):
    self.log_interval_counter += 1
    if is_satisfied_condition:
      self.__log(self.generate_finish_message(iteration_count, deltas))
    elif self.log_interval_counter >= self.log_interval:
      self.__log(self.generate_progress_message(iteration_count, deltas))
      self.log_interval_counter = 0

  def __build_err_msg(self, msg):
    base_msg = "[ {0} ] class does not implement [ {1} ] method"
    return base_msg.format(self.__class__.__name__, msg)

  def __log(self, message):
    if message and len(message) != 0:
      logging.info(message)


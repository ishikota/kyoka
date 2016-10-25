import time
from kyoka.finish_rule.base_finish_rule import BaseFinishRule

class WatchIterationCount(BaseFinishRule):

  def __init__(self, target_count, log_interval=1):
    BaseFinishRule.__init__(self, log_interval)
    self.target_count = target_count
    self.start_time = self.last_update_time = 0

  def define_log_tag(self):
    return "Progress"

  def check_condition(self, iteration_count):
    return iteration_count >= self.target_count

  def generate_start_message(self):
    self.start_time = self.last_update_time = time.time()
    return "Start GPI iteration for %d times" % self.target_count

  def generate_progress_message(self, iteration_count):
    current_time = time.time()
    msg = "Finished %d / %d iterations (%.1fs)" %\
            (iteration_count, self.target_count, current_time - self.last_update_time)
    self.last_update_time = current_time
    return msg

  def generate_finish_message(self, iteration_count):
    base_msg = "Completed GPI iteration for %d times. (total time: %ds)"
    return base_msg % (iteration_count, time.time() - self.start_time)

  def __build_err_msg(self, msg):
    base_msg = "[ {0} ] class does not implement [ {1} ] method"
    return base_msg.format(self.__class__.__name__, msg)


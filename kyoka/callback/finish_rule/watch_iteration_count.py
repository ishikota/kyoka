import time
from kyoka.callback.finish_rule.base_finish_rule import BaseFinishRule

class WatchIterationCount(BaseFinishRule):

  def __init__(self, target_count, verbose=1):
    self.target_count = target_count
    self.start_time = self.last_update_time = 0
    self.verbose = verbose

  def define_log_tag(self):
    return "Progress"

  def check_condition(self, iteration_count, domain, value_function):
    return iteration_count >= self.target_count

  def generate_start_message(self):
    self.start_time = self.last_update_time = time.time()
    return "Start GPI iteration for %d times" % self.target_count

  def generate_finish_message(self, iteration_count):
    base_msg = "Completed GPI iteration for %d times. (total time: %ds)"
    return base_msg % (iteration_count, time.time() - self.start_time)

  def before_update(self, iteration_count, domain, value_function):
    super(WatchIterationCount, self).before_update(iteration_count, domain, value_function)
    self.last_update_time = time.time()

  def after_update(self, iteration_count, domain, value_function):
    super(WatchIterationCount, self).after_update(iteration_count, domain, value_function)
    if self.verbose > 0:
      current_time = time.time()
      msg = "Finished %d / %d iterations (%.1fs)" %\
              (iteration_count, self.target_count, current_time - self.last_update_time)
      self.last_update_time = current_time
      self.log(msg)


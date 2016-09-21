from kyoka.finish_rule.base_finish_rule import BaseFinishRule

class WatchIterationCount(BaseFinishRule):

  def __init__(self, target_count, log_interval=100):
    BaseFinishRule.__init__(self, log_interval)
    self.target_count = target_count

  def check_condition(self, iteration_count, _deltas):
    return iteration_count >= self.target_count

  def generate_progress_message(self, iteration_count, deltas):
    base_msg = "Finished %d / %d iterations"
    return base_msg % (iteration_count, self.target_count)

  def generate_finish_message(self, iteration_count, deltas):
    base_msg = "Completed GPI iteration for %d times."
    return base_msg % iteration_count

  def __build_err_msg(self, msg):
    base_msg = "[ {0} ] class does not implement [ {1} ] method"
    return base_msg.format(self.__class__.__name__, msg)


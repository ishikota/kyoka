from kyoka.algorithm.finish_rule.base_finish_rule import BaseFinishRule

class WatchUpdateDelta(BaseFinishRule):

  def __init__(self, patience, minimum_required_delta, log_interval=100):
    BaseFinishRule.__init__(self, log_interval)
    self.patience = patience
    self.minimum_required_delta = minimum_required_delta
    self.no_update_counter = 0

  def check_condition(self, _iteration_count, deltas):
    max_delta = max([abs(delta) for delta in deltas])
    if max_delta >= self.minimum_required_delta:
      self.no_update_counter = 0
    else:
      self.no_update_counter += 1
    return self.no_update_counter >= self.patience

  def generate_progress_message(self, iteration_count, deltas):
    base_msg = "Current iteration count = %d, finish if no update within %d iteration."
    return base_msg % (iteration_count, self.patience - self.no_update_counter)

  def generate_finish_message(self, iteration_count, deltas):
    base_msg = "Update of deltas are less than %f while %d iteration. So stop GPI process"
    return base_msg % (self.minimum_required_delta, self.patience)


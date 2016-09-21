from kyoka.finish_rule.base_finish_rule import BaseFinishRule

class WatchUpdateDelta(BaseFinishRule):

  def __init__(self, patience, minimum_required_delta, log_interval=100):
    BaseFinishRule.__init__(self, log_interval)
    self.patience = patience
    self.minimum_required_delta = minimum_required_delta
    self.no_update_counter = 0
    self.max_delta_memo = []

  def check_condition(self, _iteration_count, deltas):
    max_delta = max([abs(delta) for delta in deltas])
    self.max_delta_memo.append(max_delta)
    if max_delta >= self.minimum_required_delta:
      self.no_update_counter = 0
      self.max_delta_memo = []
    else:
      self.no_update_counter += 1
    return self.no_update_counter >= self.patience

  def generate_progress_message(self, iteration_count, deltas):
    base_msg = "Current iteration count = %d, finish if no update within %d iteration."
    return base_msg % (iteration_count, self.patience - self.no_update_counter)

  def generate_finish_message(self, iteration_count, deltas):
    base_msg = "Update of deltas are less than %f while %d iteration (maimum delta was %f). So stop GPI process"
    maximum_delta = max(self.max_delta_memo)
    return base_msg % (self.minimum_required_delta, self.patience, maximum_delta)


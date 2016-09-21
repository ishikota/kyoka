import os
from kyoka.finish_rule.base_finish_rule import BaseFinishRule

class ManualInterruption(BaseFinishRule):

  TARGET_WARD = "stop"

  def __init__(self, monitor_file_path, log_interval=100):
    BaseFinishRule.__init__(self, log_interval)
    self.monitor_file_path = monitor_file_path

  def check_condition(self, _iteration_count, _deltas):
    return self.__order_found_in_monitoring_file(self.monitor_file_path, self.TARGET_WARD)

  def generate_progress_message(self, _iteration_count, _deltas):
    return None

  def generate_finish_message(self, iteration_count, _deltas):
    base_msg = "Interrupt GPI after %d iterations because interupption order found in [ %s ]."
    return base_msg % (iteration_count, self.monitor_file_path)

  def __order_found_in_monitoring_file(self, filepath, target_word):
    return os.path.isfile(filepath) and self.__found_target_ward_in_file(filepath, target_word)

  def __found_target_ward_in_file(self, filepath, target_word):
    search_word = lambda src, target: target in src
    src = self.__read_data(filepath)
    return search_word(src, target_word) if src else False

  def __read_data(self, filepath):
    with open(filepath, 'rb') as f:
      return f.read()


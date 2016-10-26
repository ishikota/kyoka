import os
import time
from kyoka.callback.finish_rule.base_finish_rule import BaseFinishRule

class ManualInterruption(BaseFinishRule):

  TARGET_WARD = "stop"

  def __init__(self, monitor_file_path, watch_interval=30):
    self.monitor_file_path = monitor_file_path
    self.watch_interval = watch_interval

  def check_condition(self, _iteration_count, _domain, _value_function):
    current_time = time.time()
    if current_time - self.last_check_time >= self.watch_interval:
      self.last_check_time = current_time
      return self.__order_found_in_monitoring_file(self.monitor_file_path, self.TARGET_WARD)
    else:
      return False

  def generate_start_message(self):
    self.last_check_time = time.time()
    base_first_msg ='Write word "%s" on file "%s" will finish the GPI'
    base_second_msg = "(Stopping GPI may take about %s seconds. Because we check target file every %s seconds.)"
    first_msg = base_first_msg % (self.TARGET_WARD, self.monitor_file_path)
    second_msg = base_second_msg % (self.watch_interval, self.watch_interval)
    return "\n".join([first_msg, second_msg])

  def generate_finish_message(self, iteration_count):
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


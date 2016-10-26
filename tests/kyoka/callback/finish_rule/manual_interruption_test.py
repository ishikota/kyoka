import os

from tests.base_unittest import BaseUnitTest
from kyoka.callback.finish_rule.manual_interruption import ManualInterruption

from nose.tools import raises
from mock import patch

class ManualInterruptionTest(BaseUnitTest):

  def setUp(self):
    file_path = self.__generate_tmp_file_path()
    self.rule = ManualInterruption(monitor_file_path=file_path, watch_interval=5)

  def tearDown(self):
    file_path = self.__generate_tmp_file_path()
    if os.path.isfile(file_path):
      os.remove(file_path)

  def test_define_log_tag(self):
    self.eq("ManualInterruption", self.rule.define_log_tag())

  def test_check_condition(self):
    file_path = self.__generate_tmp_file_path()
    mock_return = [1, 2, 3, 6, 7, 12]
    with patch('time.time', side_effect=mock_return):
      self.rule.generate_start_message()
      self.false(self.rule.check_condition("dummy", "dummy", "dummy"))
      self.__write_word(file_path, "hoge")
      self.false(self.rule.check_condition("dummy", "dummy", "dummy"))
      self.false(self.rule.check_condition("dummy", "dummy", "dummy"))
      self.__write_word(file_path, "stop")
      self.false(self.rule.check_condition("dummy", "dummy", "dummy"))
      self.true(self.rule.check_condition("dummy", "dummy", "dummy"))

  def test_generate_start_message(self):
    message = self.rule.generate_start_message()
    self.include(self.rule.TARGET_WARD, message)
    self.include(self.rule.monitor_file_path, message)
    self.include(str(self.rule.watch_interval), message)

  def test_generate_finish_message(self):
    file_path = self.__generate_tmp_file_path()
    msg = self.rule.generate_finish_message(5)
    self.include(str(5), msg)
    self.include(file_path, msg)

  def __generate_tmp_file_path(self):
    return os.path.join(os.path.dirname(__file__), "tmp_file_for_manual_interruption_test.tmp")

  def __write_word(self, filepath, word):
    with open(filepath, 'wb') as f:
      f.write(word)


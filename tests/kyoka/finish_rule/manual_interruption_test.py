import os

from tests.base_unittest import BaseUnitTest
from kyoka.finish_rule.manual_interruption import ManualInterruption

from nose.tools import raises

class ManualInterruptionTest(BaseUnitTest):

  def setUp(self):
    file_path = self.__generate_tmp_file_path()
    self.rule = ManualInterruption(monitor_file_path=file_path)

  def tearDown(self):
    file_path = self.__generate_tmp_file_path()
    if os.path.isfile(file_path):
      os.remove(file_path)

  def test_satisfy_condition(self):
    file_path = self.__generate_tmp_file_path()
    self.false(self.rule.satisfy_condition(1, "dummy"))
    self.__write_word(file_path, "hoge")
    self.false(self.rule.satisfy_condition(1, "dummy"))
    self.__write_word(file_path, "stop")
    self.true(self.rule.satisfy_condition(1, "dummy"))

  def test_generate_progress_message(self):
    self.eq(None, self.rule.generate_progress_message(5, "dummy"))

  def test_generate_finish_message(self):
    file_path = self.__generate_tmp_file_path()
    msg = self.rule.generate_finish_message(5, "dummy")
    self.include(str(5), msg)
    self.include(file_path, msg)

  def __generate_tmp_file_path(self):
    return os.path.join(os.path.dirname(__file__), "tmp_file_for_manual_interruption_test.tmp")

  def __write_word(self, filepath, word):
    with open(filepath, 'wb') as f:
      f.write(word)


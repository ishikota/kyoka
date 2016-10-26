from tests.base_unittest import BaseUnitTest
from kyoka.callback.finish_rule.base_finish_rule import BaseFinishRule

from nose.tools import raises
from mock import patch

import StringIO
import sys

class BaseFinishRuleTest(BaseUnitTest):

  def setUp(self):
    self.rule = BaseFinishRule()
    self.capture = StringIO.StringIO()
    sys.stdout = self.capture

  def tearDown(self):
    sys.stdout = sys.__stdout__

  @raises(NotImplementedError)
  def test_check_condition(self):
    self.rule.check_condition("dummy", "dummy", "dummy")

  @raises(NotImplementedError)
  def test_generate_start_message(self):
    self.rule.generate_start_message()

  @raises(NotImplementedError)
  def test_generate_finish_message(self):
    self.rule.generate_finish_message("dummy")

  def test_interrupt_gpi(self):
    rule = self.TestImplementation()
    self.false(rule.interrupt_gpi(1, "dummy", "dummy"))
    self.true(rule.interrupt_gpi(2, "dummy", "dummy"))
    self.false(rule.interrupt_gpi(3, "dummy", "dummy"))

  def test_log_start_message(self):
    rule = self.TestImplementation()
    rule.before_gpi_start("dummy", "dummy")
    self.eq('[test_tag] start\n', self.capture.getvalue())

  def test_log_finish_message(self):
    rule = self.TestImplementation()
    self.false(rule.interrupt_gpi(1, "domain", "value_function"))
    self.eq('', self.capture.getvalue())
    self.true(rule.interrupt_gpi(2, "domain", "value_function"))
    self.eq('[test_tag] finish:2\n', self.capture.getvalue())

  class TestImplementation(BaseFinishRule):

    def define_log_tag(self):
      return "test_tag"

    def check_condition(self, iteration_count, _domain, _value_function):
      return iteration_count == 2

    def generate_start_message(self):
      return "start"

    def generate_finish_message(self, iteration_count):
      return "%s:%s" % ("finish", iteration_count)


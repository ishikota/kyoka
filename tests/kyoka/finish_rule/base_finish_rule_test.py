from tests.base_unittest import BaseUnitTest
from kyoka.finish_rule.base_finish_rule import BaseFinishRule

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

  def test_satisfy_condition_logging(self):
    rule = self.TestImplementation(log_interval=2)
    self.false(rule.satisfy_condition(1))
    self.eq('', self.capture.getvalue())
    self.false(rule.satisfy_condition(2))
    self.eq('[test_tag] progress:2\n', self.capture.getvalue())
    self.capture = StringIO.StringIO()
    sys.stdout = self.capture
    self.false(rule.satisfy_condition(3))
    self.eq('', self.capture.getvalue())
    self.capture = StringIO.StringIO()
    sys.stdout = self.capture
    self.true(rule.satisfy_condition(0))
    self.eq('[test_tag] finish:0\n', self.capture.getvalue())

  def test_log_progress(self):
    rule = self.TestImplementation()
    rule.log_progress(1)
    self.eq('[test_tag] progress:1\n', self.capture.getvalue())

  def test_log_finish_message(self):
    rule = self.TestImplementation()
    rule.log_finish_message(1)
    self.eq('[test_tag] finish:1\n', self.capture.getvalue())

  @raises(NotImplementedError)
  def test_define_log_tag(self):
    self.rule.define_log_tag()

  @raises(NotImplementedError)
  def test_check_condition(self):
    self.rule.check_condition("dummy")

  @raises(NotImplementedError)
  def test_generate_progress_condition(self):
    self.rule.check_condition("dummy")

  @raises(NotImplementedError)
  def test_generate_finish_message(self):
    self.rule.generate_finish_message("dummy")

  class TestImplementation(BaseFinishRule):

    def define_log_tag(self):
      return "test_tag"

    def check_condition(self, iteration_count):
      return iteration_count == 0

    def generate_progress_message(self, iteration_count):
      return "%s:%s" % ("progress", iteration_count)

    def generate_finish_message(self, iteration_count):
      return "%s:%s" % ("finish", iteration_count)


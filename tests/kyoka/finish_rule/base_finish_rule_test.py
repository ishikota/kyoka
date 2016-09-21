from tests.base_unittest import BaseUnitTest
from kyoka.finish_rule.base_finish_rule import BaseFinishRule

from nose.tools import raises
from mock import patch

class BaseFinishRuleTest(BaseUnitTest):

  def setUp(self):
    self.rule = BaseFinishRule()

  def test_satisfy_condition_logging(self):
    rule = self.TestImplementation(log_interval=2)
    with patch("logging.info") as log_mock:
      self.false(rule.satisfy_condition(1, 0))
      log_mock.assert_not_called()
      self.false(rule.satisfy_condition(2, 0))
      log_mock.assert_called_with("progress:2:0")
      self.false(rule.satisfy_condition(3, 0))
      log_mock.assert_called_with("progress:2:0")
      self.true(rule.satisfy_condition(0, 0))
      log_mock.assert_called_with("finish:0:0")

  @raises(NotImplementedError)
  def test_check_condition(self):
    self.rule.check_condition("dummy", "dummy")

  @raises(NotImplementedError)
  def test_generate_progress_condition(self):
    self.rule.check_condition("dummy", "dummy")

  @raises(NotImplementedError)
  def test_generate_finish_message(self):
    self.rule.generate_finish_message("dummy", "dummy")

  class TestImplementation(BaseFinishRule):

    def check_condition(self, iteration_count, deltas):
      return iteration_count == 0

    def generate_progress_message(self, iteration_count, deltas):
      return "%s:%s:%s" % ("progress", iteration_count, deltas)

    def generate_finish_message(self, iteration_count, deltas):
      return "%s:%s:%s" % ("finish", iteration_count, deltas)


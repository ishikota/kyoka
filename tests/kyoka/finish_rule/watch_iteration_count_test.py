from tests.base_unittest import BaseUnitTest
from kyoka.finish_rule.watch_iteration_count import WatchIterationCount

from nose.tools import raises

class BaseFinishRuleTest(BaseUnitTest):

  def setUp(self):
    self.rule = WatchIterationCount(target_count=100)

  def test_satisfy_condition(self):
    self.false(self.rule.satisfy_condition(99, "dummy"))
    self.true(self.rule.satisfy_condition(100, "dummy"))
    self.true(self.rule.satisfy_condition(101, "dummy"))

  def test_generate_progress_message(self):
    msg = self.rule.generate_progress_message(5, "dummy")
    self.include(str(5), msg)
    self.include(str(100), msg)

  def test_generate_finish_message(self):
    self.include(str(5), self.rule.generate_finish_message(5, "dummy"))


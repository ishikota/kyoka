from tests.base_unittest import BaseUnitTest
from kyoka.algorithm.finish_rule.watch_update_delta import WatchUpdateDelta

from nose.tools import raises

class WatchUpdateDeltaTest(BaseUnitTest):

  def test_satisfy_condition(self):
    rule = WatchUpdateDelta(patience=2, minimum_required_delta=2)
    self.false(rule.satisfy_condition("dummy", [1]))
    self.false(rule.satisfy_condition("dummy", [1, 2]))
    self.false(rule.satisfy_condition("dummy", [-1]))
    self.false(rule.satisfy_condition("dummy", [-1, -3]))
    self.false(rule.satisfy_condition("dummy", [1, -1]))
    self.true(rule.satisfy_condition("dummy", [1.9, -1]))

  def test_generate_finish_message(self):
    rule = WatchUpdateDelta(patience=2, minimum_required_delta=3)
    rule.satisfy_condition(1, [1.9, -1])
    msg = rule.generate_finish_message(5, "dummy")
    self.include(str(2), msg)
    self.include(str(3), msg)


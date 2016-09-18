from tests.base_unittest import BaseUnitTest
from kyoka.algorithm.finish_rule.base_finish_rule import BaseFinishRule

from nose.tools import raises

class BaseFinishRuleTest(BaseUnitTest):

  def setUp(self):
    self.rule = BaseFinishRule()

  @raises(NotImplementedError)
  def test_satisfy_condition(self):
    self.rule.satisfy_condition("dummy", "dummy")

  @raises(NotImplementedError)
  def test_generate_finish_message(self):
    self.rule.generate_finish_message("dummy", "dummy")


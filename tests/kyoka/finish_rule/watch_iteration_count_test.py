from tests.base_unittest import BaseUnitTest
from kyoka.finish_rule.watch_iteration_count import WatchIterationCount

import sys
import StringIO
from nose.tools import raises
from mock import patch

class BaseFinishRuleTest(BaseUnitTest):

  def setUp(self):
    self.rule = WatchIterationCount(target_count=100)

  def tearDown(self):
    sys.stdout = sys.__stdout__

  def test_define_log_tag(self):
    self.eq("Progress", self.rule.define_log_tag())

  def test_satisfy_condition(self):
    self.false(self.rule.satisfy_condition(99))
    self.true(self.rule.satisfy_condition(100))
    self.true(self.rule.satisfy_condition(101))

  def test_generat_start_message(self):
    self.include(str(100), self.rule.generate_start_message())

  def test_generate_progress_message(self):
    msg = self.rule.generate_progress_message(5)
    self.include(str(5), msg)
    self.include(str(100), msg)

  def test_calculation_time_in_prgress_message(self):
    rule = WatchIterationCount(target_count=100, log_interval=1)
    mock_time_return = [1477394336.834469, 1477394338.815435, 1477394340.965727, 1477394343.925256]
    expected_calc_time = [1.9809658527374268, 2.150292158126831, 2.959528923034668]
    with patch('time.time', side_effect=mock_time_return):
      rule.log_start_message()
      for expected in expected_calc_time:
        capture = StringIO.StringIO()
        sys.stdout = capture
        rule.satisfy_condition(1)
        self.include("%.1f" % expected, capture.getvalue())

  def test_generate_finish_message(self):
    self.include(str(5), self.rule.generate_finish_message(5))


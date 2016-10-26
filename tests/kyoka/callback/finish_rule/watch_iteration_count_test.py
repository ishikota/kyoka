from tests.base_unittest import BaseUnitTest
from kyoka.callback.finish_rule.watch_iteration_count import WatchIterationCount

import sys
import StringIO
from nose.tools import raises
from mock import patch

class WatchIterationCountTest(BaseUnitTest):

  def setUp(self):
    self.rule = WatchIterationCount(target_count=100)

  def tearDown(self):
    sys.stdout = sys.__stdout__

  def test_define_log_tag(self):
    self.eq("Progress", self.rule.define_log_tag())

  def test_check_condition(self):
    self.false(self.rule.check_condition(99, "dummy", "dummy"))
    self.true(self.rule.check_condition(100, "dummy", "dummy"))
    self.true(self.rule.check_condition(101, "dummy", "dummy"))

  def test_generate_start_message(self):
    self.include(str(100), self.rule.generate_start_message())

  def test_generate_finish_message(self):
    with patch('time.time', side_effect=[1477394336.834469, 1477394338.815435]):
      self.rule.generate_start_message()
      message = self.rule.generate_finish_message(5)
      self.include(str(5), message)
      self.include(str(1), message)

  def test_message_for_lifecycle(self):
    rule = WatchIterationCount(target_count=2)
    mock_time_return = [
        1477462649.467603,
        1477462653.996855,
        1477462655.032283,
        1477462658.945142,
        1477462663.027276,
        1477462679.056145
    ]
    expected_calc_time = [1.0354280471801758, 4.082134008407593]
    expected_interrupt_return = [False, True]
    expected_total_time = 29.588541984558105

    with patch('time.time', side_effect=mock_time_return):
      rule.before_gpi_start("dummy", "dummy")
      for idx, expected in enumerate(expected_calc_time, start=1):
        capture = StringIO.StringIO()
        sys.stdout = capture
        rule.before_update("dummy", "dummy", "dummy")
        rule.after_update(idx, "dummy", "dummy")
        self.eq(expected_interrupt_return[idx-1], rule.interrupt_gpi(idx, "dummy", "dummy"))
        self.include("[Progress]", capture.getvalue())
        self.include("%.1f" % expected, capture.getvalue())
        self.include("%d /" % idx, capture.getvalue())
      rule.after_gpi_finish("dummy", "dummy")
      self.include("%d" % expected_total_time, capture.getvalue())

  def test_verbose_mode(self):
    capture = StringIO.StringIO()
    sys.stdout = capture
    rule = WatchIterationCount(target_count=100, verbose=0)
    rule.before_gpi_start("dummy", "dummy")
    rule.before_update(1, "dummy", "dummy")
    rule.after_update(1, "dummy", "dummy")
    rule.interrupt_gpi(1, "dummy", "dummy")
    rule.before_update(100, "dummy", "dummy")
    rule.after_update(100, "dummy", "dummy")
    rule.interrupt_gpi(100, "dummy", "dummy")
    rule.after_gpi_finish("dummy", "dummy")
    self.include("Completed", capture.getvalue())
    self.not_include("Finished", capture.getvalue())


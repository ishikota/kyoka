from tests.base_unittest import BaseUnitTest
from kyoka.callback.base_performance_watcher import BasePerformanceWatcher
from nose.tools import raises

import sys
import StringIO

class BasePerformanceWatcherTest(BaseUnitTest):

  def setUp(self):
    self.capture = StringIO.StringIO()
    sys.stdout = self.capture

  def tearDown(self):
    sys.stdout = sys.__stdout__

  @raises(NotImplementedError)
  def test_implementation_check_on_define_performance_test_interval(self):
    BasePerformanceWatcher().define_performance_test_interval()

  @raises(NotImplementedError)
  def test_implementation_check_on_run_performance_test(self):
    BasePerformanceWatcher().run_performance_test("dummy", "dummy")

  def test_default_log_message(self):
    watcher = self.TestMinimumImplementation()
    watcher.before_gpi_start("dummy", "dummy")
    watcher.after_update(1, "dummy", "dummy")
    watcher.after_update(2, "dummy", "dummy")
    expected = "[TestMinimumImplementation] Performance test result : 3 (nb_iteration=2)\n"
    self.eq(expected, self.capture.getvalue())

  def test_setup_is_called(self):
    watcher = self.TestCompleteImplementation()
    self.false(watcher.is_setup_called)
    watcher.before_gpi_start("dummy", "dummy")
    self.true(watcher.is_setup_called)

  def test_teardown_is_called(self):
    watcher = self.TestCompleteImplementation()
    self.false(watcher.is_teardown_called)
    watcher.after_gpi_finish("dummy", "dummy")
    self.true(watcher.is_teardown_called)

  def test_timing_of_performance_test(self):
    watcher = self.TestCompleteImplementation()
    watcher.before_gpi_start("dummy", "dummy")
    watcher.after_update(1, "dummy", "dummy")
    self.eq('', self.capture.getvalue())
    watcher.after_update(2, "dummy", "dummy")
    self.eq('[Test] test:1\n', self.capture.getvalue())
    watcher.after_update(3, "dummy", "dummy")
    self.eq('[Test] test:1\n', self.capture.getvalue())
    watcher.after_update(4, "dummy", "dummy")
    self.eq('[Test] test:1\n[Test] test:4\n', self.capture.getvalue())
    self.eq([1, 4], watcher.performance_log)

  class TestCompleteImplementation(BasePerformanceWatcher):

    def __init__(self):
      self.is_setup_called = False
      self.is_teardown_called = False
      self.test_count = 0

    def setUp(self, domain, value_function):
      self.is_setup_called = True

    def tearDown(self, domain, value_function):
      self.is_teardown_called = True

    def define_performance_test_interval(self):
      return 2

    def run_performance_test(self, _domain, _value_function):
      self.test_count += 1
      return self.test_count**2

    def define_log_message(self, iteration_count, domain, value_function, test_result):
      return "test:%s" % test_result

    def define_log_tag(self):
      return "Test"

  class TestMinimumImplementation(BasePerformanceWatcher):

    def define_performance_test_interval(self):
      return 2

    def run_performance_test(self, _domain, _value_function):
      return 3


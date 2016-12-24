import os
import sys
import StringIO

from nose.tools import raises
from mock import patch, Mock

from kyoka.callback import BaseCallback, BasePerformanceWatcher, EpsilonAnnealer,\
        LearningRecorder, BaseFinishRule, ManualInterruption, WatchIterationCount
from kyoka.policy import EpsilonGreedyPolicy
from tests.base_unittest import BaseUnitTest
from tests.utils import generate_tmp_dir_path, setup_tmp_dir, teardown_tmp_dir, remove_leaf_dir


def capture_log(instance):
    instance.capture = StringIO.StringIO()
    sys.stdout = instance.capture

def release_capture():
   sys.stdout = sys.__stdout__

class BaseCallbackTest(BaseUnitTest):

    def setUp(self):
        self.callback = BaseCallback()
        capture_log(self)

    def tearDown(self):
        release_capture()

    def test_interrupt_gpi(self):
        self.false(self.callback.interrupt_gpi("dummy", "dummy", "dummy"))

    def test_define_log_tag(self):
        self.eq("BaseCallback", self.callback.define_log_tag())

    def test_log(self):
        self.callback.log("hoge")
        self.eq("[BaseCallback] hoge\n", self.capture.getvalue())

class BasePerformanceWatcherTest(BaseUnitTest):

    def setUp(self):
        capture_log(self)

    def tearDown(self):
        release_capture()

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

        def setUp(self, task, value_function):
            self.is_setup_called = True

        def tearDown(self, task, value_function):
            self.is_teardown_called = True

        def define_performance_test_interval(self):
             return 2

        def run_performance_test(self, _task, _value_function):
            self.test_count += 1
            return self.test_count**2

        def define_log_message(self, iteration_count, task, value_function, test_result):
            return "test:%s" % test_result

        def define_log_tag(self):
            return "Test"

    class TestMinimumImplementation(BasePerformanceWatcher):

        def define_performance_test_interval(self):
            return 2

        def run_performance_test(self, _task, _value_function):
            return 3

class EpsilonAnnealerTest(BaseUnitTest):

    def setUp(self):
        self.policy = EpsilonGreedyPolicy()
        self.policy.set_eps_annealing(1.0, 0.1, 100)
        self.annealer = EpsilonAnnealer(self.policy)
        capture_log(self)

    def tearDown(self):
        release_capture()

    def test_define_log_tag(self):
        self.eq("EpsilonGreedyAnnealing", self.annealer.define_log_tag())

    def test_start_log(self):
        self.annealer.before_gpi_start("dummy", "dummy")
        self.include(str(1.0), self.capture.getvalue())
        self.include(str(0.1), self.capture.getvalue())

    def test_finish_log(self):
        [self.annealer.after_update(_, "dummy", "dummy") for _ in range(99)]
        self.not_include("finish", self.capture)
        self.annealer.after_update(100, "dummy", "dummy")
        self.include("finish", self.capture.getvalue())
        self.include(str(100), self.capture.getvalue())
        # Check that finish message is logged only once
        capture = StringIO.StringIO()
        sys.stdout = capture
        self.annealer.after_update(101, "dummy", "dummy")
        self.not_include("finish", capture.getvalue())

class LearningRecorderTest(BaseUnitTest):

    def setUp(self):
        self.algo = Mock()
        self.recorder = LearningRecorder(self.algo, generate_tmp_dir_path(__file__), 2)
        capture_log(self)

    def tearDown(self):
        release_capture()
        gen_dpath = lambda fname: os.path.join(generate_tmp_dir_path(__file__), fname)
        for idx in range(1, 5):
            dname = "after_%d_iteration" % idx
            remove_leaf_dir(gen_dpath(dname), [])
        remove_leaf_dir(gen_dpath("gpi_finished"), [])
        teardown_tmp_dir(__file__, [])

    def test_start_log(self):
        setup_tmp_dir(__file__)
        self.recorder.before_gpi_start("dummy", "dummy")
        self.include(str(2), self.capture.getvalue())
        self.include(generate_tmp_dir_path(__file__), self.capture.getvalue())

    @raises(Exception)
    def test_before_gpi_start(self):
        self.recorder.before_gpi_start("dummy", "dummy")

    def test_after_update(self):
        setup_tmp_dir(__file__)
        gen_dpath = lambda fname: os.path.join(generate_tmp_dir_path(__file__), fname)
        self.recorder.after_update(1, "dummy", "dummy")
        self.algo.save.assert_not_called()
        self.false(os.path.exists(gen_dpath("after_1_iteration")))
        self.recorder.after_update(2, "dummy", "dummy")
        self.true(os.path.exists(gen_dpath("after_2_iteration")))
        self.algo.save.assert_called_with(gen_dpath("after_2_iteration"))
        self.recorder.after_update(3, "dummy", "dummy")
        self.false(os.path.exists(gen_dpath("after_3_iteration")))
        self.algo.save.assert_called_with(gen_dpath("after_2_iteration"))
        self.recorder.after_update(4, "dummy", "dummy")
        self.true(os.path.exists(gen_dpath("after_4_iteration")))
        self.algo.save.assert_called_with(gen_dpath("after_4_iteration"))
        self.include(gen_dpath("after_4_iteration"), self.capture.getvalue())

    def test_after_gpi_finish(self):
        setup_tmp_dir(__file__)
        gen_dpath = lambda fname: os.path.join(generate_tmp_dir_path(__file__), fname)
        gen_dpath = lambda fname: os.path.join(generate_tmp_dir_path(__file__), fname)
        self.false(os.path.exists(gen_dpath("gpi_finished")))
        self.recorder.after_gpi_finish("dummy", "dummy")
        self.true(os.path.exists(gen_dpath("gpi_finished")))
        self.algo.save.assert_called_with(gen_dpath("gpi_finished"))

class BaseFinishRuleTest(BaseUnitTest):

    def setUp(self):
        self.rule = BaseFinishRule()
        capture_log(self)

    def tearDown(self):
        release_capture()

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
        self.false(rule.interrupt_gpi(1, "task", "value_function"))
        self.eq('', self.capture.getvalue())
        self.true(rule.interrupt_gpi(2, "task", "value_function"))
        self.eq('[test_tag] finish:2\n', self.capture.getvalue())

    class TestImplementation(BaseFinishRule):

        def define_log_tag(self):
            return "test_tag"

        def check_condition(self, iteration_count, _task, _value_function):
            return iteration_count == 2

        def generate_start_message(self):
            return "start"

        def generate_finish_message(self, iteration_count):
            return "%s:%s" % ("finish", iteration_count)

class ManualInterruptionTest(BaseUnitTest):

    def setUp(self):
        file_path = self.__generate_tmp_file_path()
        self.rule = ManualInterruption(monitor_file_path=file_path, watch_interval=5)

    def tearDown(self):
        file_path = self.__generate_tmp_file_path()
        if os.path.isfile(file_path): os.remove(file_path)

    def test_define_log_tag(self):
        self.eq("ManualInterruption", self.rule.define_log_tag())

    def test_check_condition(self):
        file_path = self.__generate_tmp_file_path()
        mock_return = [1, 2, 3, 6, 7, 12]
        with patch('time.time', side_effect=mock_return):
            self.rule.generate_start_message()
            self.false(self.rule.check_condition("dummy", "dummy", "dummy"))
            self.__write_word(file_path, "hoge")
            self.false(self.rule.check_condition("dummy", "dummy", "dummy"))
            self.false(self.rule.check_condition("dummy", "dummy", "dummy"))
            self.__write_word(file_path, "stop")
            self.false(self.rule.check_condition("dummy", "dummy", "dummy"))
            self.true(self.rule.check_condition("dummy", "dummy", "dummy"))

    def test_generate_start_message(self):
        message = self.rule.generate_start_message()
        self.include(self.rule.TARGET_WARD, message)
        self.include(self.rule.monitor_file_path, message)
        self.include(str(self.rule.watch_interval), message)

    def test_generate_finish_message(self):
        file_path = self.__generate_tmp_file_path()
        msg = self.rule.generate_finish_message(5)
        self.include(str(5), msg)
        self.include(file_path, msg)

    def __generate_tmp_file_path(self):
        return os.path.join(os.path.dirname(__file__), "tmp_file_for_manual_interruption_test.tmp")

    def __write_word(self, filepath, word):
      with open(filepath, 'wb') as f: f.write(word)

class WatchIterationCountTest(BaseUnitTest):

    def setUp(self):
        self.rule = WatchIterationCount(target_count=100)

    def tearDown(self):
        release_capture()

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


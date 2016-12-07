import os
import time

from utils import build_not_implemented_msg


class BaseCallback(object):
    """Base class for creating new GPI callback as you like.

    The following callback methods are called from BaseRLAlgorithm.run_gpi_for_an_episode
    in proper timing.

    before_gpi_start : called before start training
    before_update : called before each call of RLalgorithm.run_gpi_for_an_episode
    after_update : called after each call of RLalgorithm.run_gpi_for_an_episode
    after_gpi_finish : called when training is finished
    interrupt_gpi : called after each update is finished to judge if finish the training
    """

    def before_gpi_start(self, task, value_function):
        pass

    def before_update(self, iteration_count, task, value_function):
        pass

    def after_update(self, iteration_count, task, value_function):
        pass

    def after_gpi_finish(self, task, value_function):
        pass

    def interrupt_gpi(self, iteration_count, task, value_function):
        """Return True if you want to stop the training."""
        return False

    def define_log_tag(self):
        """Define tag string which displayed with log message.

        Ex. if you return "MyTag" here then
            self.log("some message") would output
            "[MyTag] some message" on your console
        """
        return self.__class__.__name__

    @property
    def tag(self):
        return self.define_log_tag()

    def log(self, message):
        if message and len(message) != 0:
            print "[%s] %s" % (self.tag, message)

class BasePerformanceWatcher(BaseCallback):
    """Utility class to execute some calculation with intermediate result of training.
    """

    def setUp(self, task, value_function):
        pass

    def tearDown(self, task, value_function):
        pass

    def define_performance_test_interval(self):
        """Define interval to execute "run_performance_test".

        For example, if you return 1 then "run_performance_test" is called after
        every update of training.
        """
        err_msg = build_not_implemented_msg(self, "define_performance_test_interval")
        raise NotImplementedError(err_msg)

    def run_performance_test(self, task, value_function):
        """Define some calculation to see how good current value_function works.

        Args:
            task: Task instance which is used in training
            value_function: Value function which is in the middle of training.
        """
        err_msg = build_not_implemented_msg(self, "run_performance_test")
        raise NotImplementedError(err_msg)

    def define_log_message(self, iteration_count, task, value_function, test_result):
        base_msg = "Performance test result : %s (nb_iteration=%d)"
        return base_msg % (test_result, iteration_count)


    def before_gpi_start(self, task, value_function):
        self.performance_log = []
        self.test_interval = self.define_performance_test_interval()
        self.setUp(task, value_function)

    def after_update(self, iteration_count, task, value_function):
        if iteration_count % self.test_interval == 0:
            result = self.run_performance_test(task, value_function)
            self.performance_log.append(result)
            message = self.define_log_message(iteration_count, task, value_function, result)
            self.log(message)

    def after_gpi_finish(self, task, value_function):
        self.tearDown(task, value_function)

class EpsilonAnnealer(BaseCallback):
    """Callback to decay epsiron of EpsironGreedyPolicy during training."""

    def __init__(self, epsilon_greedy_policy):
        """
        Args:
            epsilon_greedy_policy: target to execute epsilon annealing.
        """
        self.policy = epsilon_greedy_policy
        self.anneal_finished = False

    def define_log_tag(self):
        return "EpsilonGreedyAnnealing"

    def before_gpi_start(self, _task, _value_function):
        start_msg = "Anneal epsilon from %s to %s." % (self.policy.eps, self.policy.min_eps)
        self.log(start_msg)

    def after_update(self, iteration_count, _task, _value_function):
        self.policy.anneal_eps()
        if not self.anneal_finished and self.policy.eps == self.policy.min_eps:
            self.anneal_finished = True
            finish_msg = "Annealing has finished at %d iteration." % iteration_count
            self.log(finish_msg)

class LearningRecorder(BaseCallback):
    """Callback to save intermediate result of training.

    If you set
        root_save_dir_path=training_results
        save_interval=100
    After 250 iteration of training, "algorithm.save(training_results)" is called twice.
    So training_results directory would have two items like below

    training_results/after_100_iteration/...
                    /after_200_iteration/...
    """

    def __init__(self, algorithm, root_save_dir_path, save_interval):
        """
        Args:
            algorithm: the RL algorithm which will be used in training.
            root_save_dir_path: save method is executed under this path of directory.
            save_interval: interval of training to execute save method.
        """
        self.algorithm = algorithm
        self.root_save_dir_path = root_save_dir_path
        self.save_interval = save_interval

    def before_gpi_start(self, _task, _value_function):
        if not os.path.exists(self.root_save_dir_path):
            err_msg = "Directory [ %s ] not found which you passed to LearningRecorder."
            raise Exception(err_msg  % self.root_save_dir_path)
        base_msg = 'Your algorithm will be saved after each %d iteration on directory [ %s ].'
        self.log(base_msg % (self.save_interval, self.root_save_dir_path))

    def after_update(self, iteration_count, _task, _value_function):
        if iteration_count % self.save_interval == 0:
            dir_name = self.define_checkpoint_save_dir_name(iteration_count)
            save_path = os.path.join(self.root_save_dir_path, dir_name)
            os.mkdir(save_path)
            self.algorithm.save(save_path)
            base_msg = "Saved algorithm after %d iteration at [ %s ]."
            self.log(base_msg % (iteration_count, save_path))

    def after_gpi_finish(self, task, value_function):
        dir_name = self.define_finish_save_dir_name()
        save_path = os.path.join(self.root_save_dir_path, dir_name)
        os.mkdir(save_path)
        self.algorithm.save(save_path)

    def define_checkpoint_save_dir_name(self, iteration_count):
        return "after_%d_iteration" % iteration_count

    def define_finish_save_dir_name(self):
        return "gpi_finished"

class BaseFinishRule(BaseCallback):
    """Base class to define the rule to stop the training.

    Child class needs to implement following 3 methods.

    check_condition : return True if you want  to stop the training.
    generate_start_message : the string retured here is logged when start training.
    generate_finish_message : the string returned here is logged after training finished.
    """

    def check_condition(self, iteration_count, task, value_function):
        """Return True if you want to stop the training"""
        err_msg = build_not_implemented_msg(self, "check_condition")
        raise NotImplementedError(err_msg)

    def generate_start_message(self):
        err_msg = build_not_implemented_msg(self, "generate_start_message")
        raise NotImplementedError(err_msg)

    def generate_finish_message(self, iteration_count):
        err_msg = build_not_implemented_msg(self, "generate_finish_message")
        raise NotImplementedError(err_msg)

    def before_gpi_start(self, task, value_function):
        self.log(self.generate_start_message())

    def interrupt_gpi(self, iteration_count, task, value_function):
        finish_iteration = self.check_condition(iteration_count, task, value_function)
        if finish_iteration: self.log(self.generate_finish_message(iteration_count))
        return finish_iteration

class ManualInterruption(BaseFinishRule):
    """Callback to stop the training by file-base communication

    How to stop the training manually
    1. Share the path of a file with this callback
    2. Start the training with this callback.
    3. Write target word (default is "stop") on the shared file.
    4. Callback will find the target word from the shared file and stop the training.
    """

    TARGET_WARD = "stop"

    def __init__(self, monitor_file_path, watch_interval=30):
        """
        Args:
            monitor_file_path: path of file to send stop command.
            watch_interval: the interval to check the file in monitor_file_path
        """
        self.monitor_file_path = monitor_file_path
        self.watch_interval = watch_interval

    def check_condition(self, _iteration_count, _task, _value_function):
        current_time = time.time()
        if current_time - self.last_check_time >= self.watch_interval:
            self.last_check_time = current_time
            return self.__order_found_in_monitoring_file(self.monitor_file_path, self.TARGET_WARD)
        else:
            return False

    def generate_start_message(self):
        self.last_check_time = time.time()
        base_first_msg ='Write word "%s" on file "%s" will finish the GPI'
        base_second_msg = "(Stopping GPI may take about %s seconds. Because we check target file every %s seconds.)"
        first_msg = base_first_msg % (self.TARGET_WARD, self.monitor_file_path)
        second_msg = base_second_msg % (self.watch_interval, self.watch_interval)
        return "\n".join([first_msg, second_msg])

    def generate_finish_message(self, iteration_count):
        base_msg = "Interrupt GPI after %d iterations because interupption order found in [ %s ]."
        return base_msg % (iteration_count, self.monitor_file_path)

    def __order_found_in_monitoring_file(self, filepath, target_word):
        return os.path.isfile(filepath) and self.__found_target_ward_in_file(filepath, target_word)

    def __found_target_ward_in_file(self, filepath, target_word):
        search_word = lambda src, target: target in src
        src = self.__read_data(filepath)
        return search_word(src, target_word) if src else False

    def __read_data(self, filepath):
        with open(filepath, 'rb') as f: return f.read()

class WatchIterationCount(BaseFinishRule):
    """Finish the training after specified number of training."""

    def __init__(self, target_count, verbose=1):
        """
        Args
            target_count : stop the training after this number of iteration.
            verbose : if verbose > 0 then log is activated.
        """
        self.target_count = target_count
        self.start_time = self.last_update_time = 0
        self.verbose = verbose

    def define_log_tag(self):
        return "Progress"

    def check_condition(self, iteration_count, task, value_function):
        return iteration_count >= self.target_count

    def generate_start_message(self):
        self.start_time = self.last_update_time = time.time()
        return "Start GPI iteration for %d times" % self.target_count

    def generate_finish_message(self, iteration_count):
        base_msg = "Completed GPI iteration for %d times. (total time: %ds)"
        return base_msg % (iteration_count, time.time() - self.start_time)

    def before_update(self, iteration_count, task, value_function):
        super(WatchIterationCount, self).before_update(iteration_count, task, value_function)
        self.last_update_time = time.time()

    def after_update(self, iteration_count, task, value_function):
        super(WatchIterationCount, self).after_update(iteration_count, task, value_function)
        if self.verbose > 0:
            current_time = time.time()
            msg = "Finished %d / %d iterations (%.1fs)" % (
                    iteration_count, self.target_count,
                    current_time - self.last_update_time)
            self.last_update_time = current_time
            self.log(msg)


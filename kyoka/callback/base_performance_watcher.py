from kyoka.callback.base_callback import BaseCallback

class BasePerformanceWatcher(BaseCallback):

  def setUp(self, domain, value_function):
    pass

  def tearDown(self, domain, value_function):
    pass

  def define_performance_test_interval(self):
    err_msg = self.__build_err_msg("define_performance_test_interval")
    raise NotImplementedError(err_msg)

  def run_performance_test(self, domain, value_function):
    err_msg = self.__build_err_msg("run_performance_test")
    raise NotImplementedError(err_msg)

  def define_log_message(self, iteration_count, domain, value_function, test_result):
    base_msg = "Performance test result : %s (nb_iteration=%d)"
    return base_msg % (test_result, iteration_count)


  def before_gpi_start(self, domain, value_function):
    self.performance_log = []
    self.test_interval = self.define_performance_test_interval()
    self.setUp(domain, value_function)

  def after_update(self, iteration_count, domain, value_function):
    if iteration_count % self.test_interval == 0:
      result = self.run_performance_test(domain, value_function)
      self.performance_log.append(result)
      message = self.define_log_message(iteration_count, domain, value_function, result)
      self.log(message)

  def after_gpi_finish(self, domain, value_function):
    self.tearDown(domain, value_function)


  def __build_err_msg(self, msg):
    base_msg = "[ {0} ] class does not implement [ {1} ] method"
    return base_msg.format(self.__class__.__name__, msg)


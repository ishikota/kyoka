from kyoka.callback.base_callback import BaseCallback

class BaseFinishRule(BaseCallback):

  def check_condition(self, iteration_count, domain, value_function):
    err_msg = self.__build_err_msg("check_condition")
    raise NotImplementedError(err_msg)

  def generate_start_message(self):
    err_msg = self.__build_err_msg("generate_start_message")
    raise NotImplementedError(err_msg)

  def generate_finish_message(self, iteration_count):
    err_msg = self.__build_err_msg("generate_finish_message")
    raise NotImplementedError(err_msg)

  def before_gpi_start(self, domain, value_function):
    self.log(self.generate_start_message())

  def interrupt_gpi(self, iteration_count, domain, value_function):
    finish_iteration = self.check_condition(iteration_count, domain, value_function)
    if finish_iteration: self.log(self.generate_finish_message(iteration_count))
    return finish_iteration


  def __build_err_msg(self, msg):
    base_msg = "[ {0} ] class does not implement [ {1} ] method"
    return base_msg.format(self.__class__.__name__, msg)


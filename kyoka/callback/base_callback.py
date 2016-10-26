class BaseCallback(object):

  def before_gpi_start(self, domain, value_function):
    pass

  def before_update(self, iteration_count, domain, value_function):
    pass

  def after_update(self, iteration_count, domain, value_function):
    pass

  def after_gpi_finish(self, domain, value_function):
    pass

  def interrupt_gpi(self, iteration_count, domain, value_function):
    return False

  def define_log_tag(self):
    return self.__class__.__name__

  def log(self, message):
    if message and len(message) != 0:
      print "[%s] %s" % (self.define_log_tag(), message)


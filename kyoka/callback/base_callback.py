class BaseCallback(object):

  def before_gpi_start(self, domain, value_function):
    pass

  def before_update(self, iteration_count, domain, value_function):
    pass

  def after_update(self, iteration_count, domain, value_function):
    pass

  def after_gpi_finish(self, domain, value_function):
    pass


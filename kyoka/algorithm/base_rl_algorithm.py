class BaseRLAlgorithm(object):

  def training(self, domain, policy):
    err_msg = self.__build_err_msg("training")
    raise NotImplementedError(err_msg)

  def select_action(self, state, action):
    err_msg = self.__build_err_msg("select_action")
    raise NotImplementedError(err_msg)

  def save_value_function(self, Q):
    err_msg = self.__build_err_msg("save_value_function")
    raise NotImplementedError(err_msg)

  def load_value_function(self, Q):
    err_msg = self.__build_err_msg("load_value_function")
    raise NotImplementedError(err_msg)


  def __build_err_msg(self, msg):
    return "Accessed [ {0} ] method of BaseRLAlgorithm which should be overridden".format(msg)


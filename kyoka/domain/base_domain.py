class BaseDomain:

  def __init__(self):
    pass

  def generate_inital_state(self):
    err_msg = self.__build_err_msg("generate_inital_state")
    raise NotImplementedError(err_msg)

  def generate_initial_Q_table(self):
    err_msg = self.__build_err_msg("generate_initial_Q_table")
    raise NotImplementedError(err_msg)

  def is_terminal_state(self, state):
    err_msg = self.__build_err_msg("is_terminal_state")
    raise NotImplementedError(err_msg)

  def transit_state(self, state, action):
    err_msg = self.__build_err_msg("transit_state")
    raise NotImplementedError(err_msg)

  def generate_possible_actions(self, state):
    err_msg = self.__build_err_msg("generate_possible_actions")
    raise NotImplementedError(err_msg)

  def calculate_reward(self, state):
    err_msg = self.__build_err_msg("calculate_reward")
    raise NotImplementedError(err_msg)

  def fetch_Q_value(self, Q_table, state, action):
    err_msg = self.__build_err_msg("fetch_Q_value")
    raise NotImplementedError(err_msg)

  def update_Q_value(self, Q_table, state, action ,new_value):
    err_msg = self.__build_err_msg("update_Q_value")
    raise NotImplementedError(err_msg)


  def __build_err_msg(self, msg):
    return "Your client does not implement [ {0} ] method".format(msg)


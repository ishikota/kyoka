from kyoka.value_function.base_action_value_function import BaseActionValueFunction

class BaseDeepQLearningActionValueFunction(BaseActionValueFunction):

  def initialize_network(self):
    err_msg = self.__build_not_implemented_message("initialize_network")
    raise NotImplementedError(err_msg)

  def deepcopy_network(self, q_network):
    err_msg = self.__build_not_implemented_message("deepcopy_network")
    raise NotImplementedError(err_msg)

  def preprocess_state(self, state):
    err_msg = self.__build_not_implemented_message("preprocess_state")
    raise NotImplementedError(err_msg)

  def predict_action_value(self, network, processed_state, action):
    err_msg = self.__build_not_implemented_message("predict_action_value")
    raise NotImplementedError(err_msg)

  def train_on_minibatch(self, q_network, learning_minibatch):
    err_msg = self.__build_not_implemented_message("train_on_minibatch")
    raise NotImplementedError(err_msg)

  def save_model_weights(self, file_path):
    err_msg = self.__build_not_implemented_message("save_model_weights")
    raise NotImplementedError(err_msg)

  def load_model_weights(self, file_path):
    err_msg = self.__build_not_implemented_message("load_model_weights")
    raise NotImplementedError(err_msg)

  def update_function(self, state, action, new_value):
    err_msg = self.__build_illegal_method_access_message("update_function")
    raise AttributeError(err_msg)


  def setUp(self):
    self.Q = self.initialize_network()
    self.sync_target_network()
    self.use_target_network_flg = False

  def use_target_network(self, use_target_net):
    self.use_target_network_flg = use_target_net

  def calculate_value(self, state, action):
    if self.use_target_network_flg:
      return self.predict_action_value(self.Q_hat, state, action)
    else:
      return self.predict_action_value(self.Q, state, action)

  def sync_target_network(self):
    self.Q_hat = self.deepcopy_network(self.Q)

  def provide_data_to_store(self):
    return None

  def receive_data_to_restore(self, restored_data):
    pass

  def __build_not_implemented_message(self, methodname):
    base_msg = "[ {0} ] class does not implement [ {1} ] method"
    return base_msg.format(self.__class__.__name__, methodname)

  def __build_illegal_method_access_message(self, methodname):
    base_msg = '"%s" is called on DQNActionValueFunction. \n' +\
            '(Maybe you are using DQNActionValueFunction for not DQN algorithm like "QLearning") '
    return base_msg % methodname


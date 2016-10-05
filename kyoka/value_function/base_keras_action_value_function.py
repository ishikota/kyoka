from kyoka.value_function.base_action_value_function import BaseActionValueFunction
import numpy as np

class BaseKerasActionValueFunction(BaseActionValueFunction):

  def setUp(self):
    self.model = self.generate_model()
    self.last_metric = 0

  def calculate_value(self, state, action):
    model_input = self.transform_state_action_into_input(state, action)
    return self.predict_value(self.model, model_input)

  def update_function(self, state, action, new_value):
    model_input = self.transform_state_action_into_input(state, action)
    X, Y = map(self.__wrap_data_for_batch, [model_input, new_value])
    metrics = self.model.train_on_batch(X, Y)
    metric = self.fetch_training_metric(metrics)
    delta = metric - self.last_metric
    self.last_metric = metric
    return delta

  def save_model_weights(self, file_path):
    self.model.save_weights(file_path)

  def load_model_weights(self, file_path):
    self.model.load_weights(file_path)

  def provide_data_to_store(self):
    pass

  def receive_data_to_restore(self, restored_data):
    pass

  def generate_model(self):
    err_msg = self.__build_err_msg("generate_model")
    raise NotImplementedError(err_msg)

  def transform_state_action_into_input(self, state, action):
    err_msg = self.__build_err_msg("transform_state_action_into_input")
    raise NotImplementedError(err_msg)

  def predict_value(self, model, X):
    err_msg = self.__build_err_msg("predict_value")
    raise NotImplementedError(err_msg)

  def fetch_training_metric(self, metrics):
    err_msg = self.__build_err_msg("fetch_training_metric")
    raise NotImplementedError(err_msg)

  def __wrap_data_for_batch(self, data):
    return np.array([data])

  def __build_err_msg(self, msg):
    base_msg = "[ {0} ] class does not implement [ {1} ] method"
    return base_msg.format(self.__class__.__name__, msg)


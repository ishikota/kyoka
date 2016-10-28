from kyoka.value_function.base_deep_q_learning_action_value_function import BaseDeepQLearningActionValueFunction
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import os
import numpy as np

class MazeDQNValueFunction(BaseDeepQLearningActionValueFunction):

  Q_NETWORK_SAVE_FILE_NAME = "maze_q_network_weights.h5"
  Q_HAT_NETWORK_SAVE_FILE_NAME = "maze_q_hat_network_weights.h5"
  TMP_FILE_NAME = "maze_dqn_deepcopy_tmp.h5"

  def __init__(self, domain):
    BaseDeepQLearningActionValueFunction.__init__(self)
    self.domain = domain
    self.maze_shape = domain.get_maze_shape()

  def initialize_network(self):
    model = self.__gen_model()
    model.compile(loss="mse",  optimizer="adam")
    return model

  def deepcopy_network(self, q_network):
    tmp_file_path = self.__gen_tmp_weight_file_path()
    q_network.save_weights(tmp_file_path, overwrite=True)
    target_network = self.__gen_model()
    target_network.load_weights(tmp_file_path)
    os.remove(tmp_file_path)
    return target_network

  def preprocess_state_sequence(self, raw_state_sequence):
    return raw_state_sequence[-1]

  def predict_action_value(self, q_network, processed_state, action):
    X = self.__transform_state_action_into_input(processed_state, action)
    return q_network.predict(np.array([X]))[0][0]

  def train_on_minibatch(self, q_network, learning_minibatch):
    processed_minibatch = [(self.__transform_state_action_into_input(state, action), target)\
            for state, action, target in learning_minibatch]
    X = np.array([x for x, _ in processed_minibatch])
    y = np.array([y for _, y in processed_minibatch])
    history = self.Q.fit(X, y, batch_size=1, nb_epoch=1, shuffle=False)

  def save_networks(self, Q_network, Q_hat_network, save_dir_path):
    Q_network.save_weights(os.path.join(save_dir_path, self.Q_NETWORK_SAVE_FILE_NAME), overwrite=True)
    Q_hat_network.save_weights(os.path.join(save_dir_path, self.Q_HAT_NETWORK_SAVE_FILE_NAME), overwrite=True)

  def load_networks(self, load_dir_path):
    q_network = self.__gen_model()
    q_hat_network = self.__gen_model()
    q_network.load_weights(os.path.join(load_dir_path, self.Q_NETWORK_SAVE_FILE_NAME))
    q_hat_network.load_weights(os.path.join(load_dir_path, self.Q_HAT_NETWORK_SAVE_FILE_NAME))
    return q_network, q_hat_network


  def __gen_model(self):
    input_dim = self.maze_shape[0] * self.maze_shape[1]
    model = Sequential()
    model.add(Dense(100, input_dim=input_dim))
    model.add(Activation("relu"))
    model.add(Dense(1))
    return model

  def __transform_state_action_into_input(self, state, action):
    flatten = lambda l: [item for sublist in l for item in sublist]
    next_state = self.domain.transit_state(state, action)
    onehot = [[1 if next_state == (row, col) else 0 for col in range(self.maze_shape[1])]\
        for row in range(self.maze_shape[0])]
    return flatten(onehot)

  def __gen_tmp_weight_file_path(self):
    return os.path.join(os.path.dirname(__file__), self.TMP_FILE_NAME)


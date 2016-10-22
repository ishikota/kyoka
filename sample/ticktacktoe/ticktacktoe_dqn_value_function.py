from sample.ticktacktoe.ticktacktoe_domain import TickTackToeDomain
from kyoka.value_function.base_deep_q_learning_action_value_function import BaseDeepQLearningActionValueFunction
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import os
import numpy as np

class TickTackToeDQNValueFunction(BaseDeepQLearningActionValueFunction):

  def __init__(self):
    BaseDeepQLearningActionValueFunction.__init__(self)
    self.domain = TickTackToeDomain()

  def initialize_network(self):
    model = self.__gen_model()
    model.compile(loss="mse",  optimizer="adam")
    return model

  def deepcopy_network(self, q_network):
    q_network.save_weights(self.__gen_tmp_weight_file_path())
    target_network = self.__gen_model()
    target_network.load_weights(self.__gen_tmp_weight_file_path())
    os.remove(self.__gen_tmp_weight_file_path())
    return target_network

  def preprocess_state(self, state):
    return state

  def predict_action_value(self, q_network, processed_state, action):
    X = self.__transform_state_action_into_input(processed_state, action)
    return q_network.predict(np.array([X]))[0][0]

  def train_on_minibatch(self, q_network, learning_minibatch):
    processed_minibatch = [(self.__transform_state_action_into_input(state, action), target)\
            for state, action, target in learning_minibatch]
    X = np.array([x for x, _ in processed_minibatch])
    y = np.array([y for _, y in processed_minibatch])
    history = self.Q.fit(X, y, batch_size=1, nb_epoch=1, shuffle=False)

  def save_model_weights(self, file_path):
    self.Q.save_weights(file_path)

  def load_model_weights(self, file_path):
    self.Q.load_weights(file_path)


  def __gen_model(self):
    model = Sequential()
    model.add(Dense(100, input_shape=(18,)))
    model.add(Activation("relu"))
    model.add(Dense(1))
    return model

  def __gen_tmp_weight_file_path(self):
    return os.path.join(os.path.dirname(__file__),\
            "ticktacktoe_dqn_value_function_test_tmp_weight.h5")

  def __transform_state_action_into_input(self, state, action):
    next_state = self.domain.transit_state(state, action)
    flg_to_ary = lambda flg: reduce(lambda acc, e: acc + [1 if (flg>>e)&1==1 else 0], range(9), [])
    multi_dim_ary = [flg_to_ary(player_board) for player_board in next_state]
    return multi_dim_ary[0] + multi_dim_ary[1]


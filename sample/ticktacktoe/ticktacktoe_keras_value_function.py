from sample.ticktacktoe.ticktacktoe_domain import TickTackToeDomain
from kyoka.value_function.base_keras_action_value_function import BaseKerasActionValueFunction
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import numpy as np

class TickTackToeKerasValueFunction(BaseKerasActionValueFunction):

  def generate_model(self):
    model = Sequential()
    model.add(Dense(100, input_shape=(18,)))
    model.add(Activation("relu"))
    model.add(Dense(1))
    model.compile(loss="mse",  optimizer="adam")
    return model

  def transform_state_action_into_input(self, state, action):
    domain = TickTackToeDomain()
    next_state = domain.transit_state(state, action)
    flg_to_ary = lambda flg: reduce(lambda acc, e: acc + [1 if (flg>>e)&1==1 else 0], range(9), [])
    multi_dim_ary = [flg_to_ary(player_board) for player_board in next_state]
    return multi_dim_ary[0] + multi_dim_ary[1]

  def fetch_training_metric(self, metrics):
    return metrics

  def predict_value(self, model, X):
    return model.predict(np.array([X]))[0][0]


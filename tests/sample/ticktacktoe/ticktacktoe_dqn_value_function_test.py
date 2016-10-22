import os
from tests.base_unittest import BaseUnitTest
from sample.ticktacktoe.ticktacktoe_dqn_value_function import TickTackToeDQNValueFunction

class TickTackToeDQNValueFunctionTest(BaseUnitTest):

  def setUp(self):
    self.func = TickTackToeDQNValueFunction()
    self.func.setUp()

  def test_transform_state_action_into_input(self):
    state = self.__gen_state("000000100", "000000010")
    action = 1
    model_input = self.func._TickTackToeDQNValueFunction__transform_state_action_into_input(state, action)
    self.eq([1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0], model_input)

  def xtest_prediction(self):
    state = self.__gen_state("000000100", "000000010")
    action = 1
    target = 5
    learning_minibatch = [(state, action, target) for i in range(32)]
    self.func.train_on_minibatch(self.func.Q, learning_minibatch)
    trained_prediction = self.func.calculate_value(state, action)
    self.func.use_target_network(True)
    target_prediction = self.func.calculate_value(state, action)
    self.neq(trained_prediction, target_prediction)

  def xtest_deepcopy(self):
    state = self.__gen_state("000000100", "000000010")
    action = 1
    target = 5
    learning_minibatch = [(state, action, target) for i in range(32)]
    self.func.train_on_minibatch(self.func.Q, learning_minibatch)
    self.func.reset_target_network()
    trained_prediction = self.func.calculate_value(state, action)
    self.func.use_target_network(True)
    target_prediction = self.func.calculate_value(state, action)
    self.almosteq(trained_prediction, target_prediction, 0.0001)

  def __gen_state(self, first_player_board, second_player_board):
    bin2i = lambda b: int(b, 2)
    return (bin2i(first_player_board), bin2i(second_player_board))


import os
from tests.base_unittest import BaseUnitTest
#from sample.ticktacktoe.ticktacktoe_dqn_value_function import TickTackToeDQNValueFunction

class TickTackToeDQNValueFunctionTest(BaseUnitTest):

  def xsetUp(self):
    self.func = TickTackToeDQNValueFunction()
    self.func.setUp()

  def xtearDown(self):
    dir_path = self.__generate_tmp_dir_path()
    file1_path = os.path.join(dir_path, "ticktacktoe_q_network_weights.h5")
    file2_path = os.path.join(dir_path, "ticktacktoe_q_hat_network_weights.h5")
    if os.path.exists(dir_path):
      if os.path.exists(file1_path):
        os.remove(file1_path)
      if os.path.exists(file2_path):
        os.remove(file2_path)
      os.rmdir(dir_path)

  def xtest_transform_state_action_into_input(self):
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

  def xtest_save_and_load(self):
    os.mkdir(self.__generate_tmp_dir_path())
    state, action, target = self.__gen_state("000000100", "000000010"), 1, 5
    learning_minibatch = [(state, action, target) for i in range(32)]
    self.func.train_on_minibatch(self.func.Q, learning_minibatch)
    self.func.save(self.__generate_tmp_dir_path())

    new_func = TickTackToeDQNValueFunction()
    new_func.setUp()
    new_func.load(self.__generate_tmp_dir_path())

    calc_target = lambda func: func.calculate_value(state, action)
    self.almosteq(calc_target(self.func), calc_target(new_func), 0.0001)

    self.func.use_target_network(True)
    new_func.use_target_network(True)
    self.almosteq(calc_target(self.func), calc_target(new_func), 0.0001)

  def __gen_state(self, first_player_board, second_player_board):
    bin2i = lambda b: int(b, 2)
    return (bin2i(first_player_board), bin2i(second_player_board))

  def __generate_tmp_dir_path(self):
    return os.path.join(os.path.dirname(__file__), "tmp")


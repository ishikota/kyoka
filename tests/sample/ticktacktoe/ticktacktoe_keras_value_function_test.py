from tests.base_unittest import BaseUnitTest
#from sample.ticktacktoe.ticktacktoe_keras_value_function import TickTackToeKerasValueFunction

class TickTackToeKerasValueFunctioneXtest(BaseUnitTest):

  def setUp(self):
    self.func = TickTackToeKerasValueFunction()
    self.func.setUp()

  def xtest_state_action_into_input(self):
    bin2i = lambda b: int(b, 2)
    first_player_board = bin2i("000000100")
    second_player_board = bin2i("000000010")
    action = 1
    model_input = self.func.transform_state_action_into_input((first_player_board, second_player_board), action)
    self.eq([1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0], model_input)

  def xtest_prediction(self):
    bin2i = lambda b: int(b, 2)
    first_player_board = bin2i("000000100")
    second_player_board = bin2i("000000010")
    state = (first_player_board, second_player_board)
    action = 1
    model_input = self.func.transform_state_action_into_input((first_player_board, second_player_board), action)
    delta = self.func.update_function(state, action, 0)
    res = self.func.calculate_value(state, action)
    self.debug()


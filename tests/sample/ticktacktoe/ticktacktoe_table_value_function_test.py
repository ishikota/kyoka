from tests.base_unittest import BaseUnitTest
from sample.ticktacktoe.ticktacktoe_table_value_function import TickTackToeTableValueFunction

class TickTackToeTableValueFunctionTest(BaseUnitTest):

  def setUp(self):
    self.func = TickTackToeTableValueFunction()
    self.func.setUp()

  def test_generate_initial_table(self):
    table = self.func.generate_initial_table()
    self.eq(2**9, len(table))
    self.eq(2**9, len(table[0]))
    self.eq(9, len(table[0][0]))
    self.eq(0, table[0][0][0])

  def test_calculate_value(self):
    bin2i = lambda b: int(b, 2)
    first_player_board = bin2i("000000100")
    second_player_board = bin2i("000000010")
    state = (first_player_board, second_player_board)
    action = 1
    self.func.table[4][2][0] = 1  # 4 = first_board, 2 = second_board, 0 = log(action, base=2)
    self.eq(1, self.func.calculate_value(state, action))

  def test_update_function(self):
    bin2i = lambda b: int(b, 2)
    first_player_board = bin2i("000000100")
    second_player_board = bin2i("000000010")
    state = (first_player_board, second_player_board)
    action = 1

    self.eq(0, self.func.calculate_value(state, action))

    self.func.update_function(state, action, 1)
    self.eq(1, self.func.calculate_value(state, action))

    self.func.update_function(state, action, 0)
    self.eq(0, self.func.calculate_value(state, action))


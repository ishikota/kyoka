from tests.base_unittest import BaseUnitTest
from sample.ticktacktoe.ticktacktoe_helper import TickTackToeHelper

class TickTackToeHelperTest(BaseUnitTest):

  def test_visualize_board_when_empty(self):
    bin2i = lambda b: int(b, 2)
    first_player_board = bin2i("000000000")
    second_player_board = bin2i("000000000")
    state = (first_player_board, second_player_board)
    expected = "- - -\n- - -\n- - -"
    self.eq(expected, TickTackToeHelper.visualize_board(state))

  def test_visualize_board_when_not_empty(self):
    bin2i = lambda b: int(b, 2)
    first_player_board = bin2i("110000000")
    second_player_board = bin2i("000000011")
    state = (first_player_board, second_player_board)
    expected = "O O -\n- - -\n- X X"
    self.eq(expected, TickTackToeHelper.visualize_board(state))


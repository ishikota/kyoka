import examples.ticktacktoe.helper as H
from examples.ticktacktoe.task import TickTackToeTask
from tests.base_unittest import BaseUnitTest


class TickTackToeHelperTest(BaseUnitTest):

    def test_visualize_board_when_empty(self):
        bin2i = lambda b: int(b, 2)
        first_player_board = bin2i("000000000")
        second_player_board = bin2i("000000000")
        state = (first_player_board, second_player_board)
        expected = "- - -\n- - -\n- - -"
        self.eq(expected, H.visualize_board(state))

    def test_visualize_board_when_not_empty(self):
        bin2i = lambda b: int(b, 2)
        first_player_board = bin2i("110000000")
        second_player_board = bin2i("000000011")
        state = (first_player_board, second_player_board)
        expected = "O O -\n- - -\n- X X"
        self.eq(expected, H.visualize_board(state))

    def test_feature_construction(self):
        bin2i = lambda b: int(b, 2)
        first_player_board = bin2i("000000100")
        second_player_board = bin2i("000000010")
        state = (first_player_board, second_player_board)
        action = 1
        features = H.construct_features(TickTackToeTask(), state, action)
        self.eq(18, len(features))
        self.eq([1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0], features)


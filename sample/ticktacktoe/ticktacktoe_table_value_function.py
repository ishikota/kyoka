import math
from kyoka.algorithm.value_function.base_action_value_function import BaseActionValueFunction

class TickTackToeTableValueFunction(BaseActionValueFunction):

  def calculate_value(self, state, action):
    first_player_board, second_player_board = state
    move_position = int(math.log(action,2))
    Q_value = self.Q_table[first_player_board][second_player_board][move_position]
    return Q_value

  def update_function(self, state, action, new_value):
    first_player_board, second_player_board = state
    move_position = int(math.log(action,2))
    delta = new_value - self.Q_table[first_player_board][second_player_board][move_position]
    self.Q_table[first_player_board][second_player_board][move_position] = new_value
    return delta

  def setUp(self):
    self.Q_table = self.__generate_initial_Q_table()


  def __generate_initial_Q_table(self):
    board_state_num = 2**9
    action_num = 9
    Q_table = [[[0 for a in range(action_num)]\
        for j in range(board_state_num)] for i in range(board_state_num)]
    return Q_table


  def deepcopy(self):
    return self

  def save(self, dest_file_path):
    pass

  def load(self, src_file_path):
    pass


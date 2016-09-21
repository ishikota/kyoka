import math
from kyoka.value_function.base_table_action_value_function import BaseTableActionValueFunction

class TickTackToeTableValueFunction(BaseTableActionValueFunction):

  def generate_initial_table(self):
    board_state_num = 2**9
    action_num = 9
    Q_table = [[[0 for a in range(action_num)]\
        for j in range(board_state_num)] for i in range(board_state_num)]
    return Q_table

  def fetch_value_from_table(self, table, state, action):
    first_player_board, second_player_board = state
    move_position = int(math.log(action,2))
    Q_value = table[first_player_board][second_player_board][move_position]
    return Q_value

  def update_table(self, table, state, action, new_value):
    first_player_board, second_player_board = state
    move_position = int(math.log(action,2))
    table[first_player_board][second_player_board][move_position] = new_value
    return table


import math
from kyoka.value_function.base_table_action_value_function import BaseTableActionValueFunction

class MazeTableValueFunction(BaseTableActionValueFunction):

  def __init__(self, maze_shape):
    BaseTableActionValueFunction.__init__(self)
    self.maze_shape = maze_shape

  def generate_initial_table(self):
    height, width = self.maze_shape
    action_num = 4
    Q_table = [[[0 for a in range(action_num)] for j in range(width)] for i in range(height)]
    return Q_table

  def fetch_value_from_table(self, table, state, action):
    row, col = state
    Q_value = table[row][col][action]
    return Q_value

  def update_table(self, table, state, action, new_value):
    row, col = state
    table[row][col][action] = new_value
    return table


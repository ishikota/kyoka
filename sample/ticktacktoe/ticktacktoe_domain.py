import math
from kyoka.domain.base_domain import BaseDomain

class TickTackToeDomain(BaseDomain):

  REWARD_WIN = 1
  REWARD_LOSE = -1
  REWARD_NONE = 0

  def __init__(self, is_first_player=True):
    self.is_first_player = is_first_player

  def generate_initial_state(self):
    first_player_board = 0
    second_player_board = 0
    state = (first_player_board, second_player_board)
    return state

  def is_terminal_state(self, state):
    return any(map(self.__is_winning, state) + [self.__is_draw(state)])

  def transit_state(self, state, action):
    first_player_board ,second_player_board = state
    count_move = lambda : bin(first_player_board | second_player_board).count("1")
    if count_move()%2==0:
      first_player_board |= action
    else:
      second_player_board |= action
    next_state = (first_player_board, second_player_board)
    return next_state

  def generate_possible_actions(self, state):
    board = state[0] | state[1]
    def add(ary, pos):
      if (board >> pos)&1 == 0:
        ary.append(1<<pos)
      return ary
    return reduce(add, range(9), [])

  def calculate_reward(self, state):
    first_player_board, second_player_board = state
    reward = self.REWARD_NONE
    if (self.__is_winning(first_player_board)):
      reward = self.REWARD_WIN
    elif(self.__is_winning(second_player_board)):
      reward = self.REWARD_LOSE
    return reward if self.is_first_player else -reward


  def __is_winning(self, player_board):
    bin2i = lambda b: int(b, 2)
    line_horizon = any([player_board & mask == mask for mask in map(bin2i, ['000000111', '000111000', '111000000'])])
    line_vertical = any([player_board & mask == mask for mask in map(bin2i, ['001001001', '010010010', '100100100'])])
    line_diagonal = any([player_board & mask == mask for mask in map(bin2i, ['100010001', '001010100'])])
    return line_horizon | line_vertical | line_diagonal

  def __is_draw(self, state):
    return len(self.generate_possible_actions(state)) == 0

  def __find_connected_line(self, player_board):
    bin2i = lambda b: int(b, 2)
    line_horizon = any([player_board & mask == mask for mask in map(bin2i, ['000000111', '000111000', '111000000'])])
    line_vertical = any([player_board & mask == mask for mask in map(bin2i, ['001001001', '010010010', '100100100'])])
    line_diagonal = any([player_board & mask == mask for mask in map(bin2i, ['100010001', '001010100'])])
    return line_horizon | line_vertical | line_diagonal


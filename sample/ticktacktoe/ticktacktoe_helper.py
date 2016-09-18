class TickTackToeHelper:

  @classmethod
  def visualize_board(self, state):
    visualize_move = lambda flg: {-1: "-", 0:"O", 1:"X"}[flg]
    fetch_line = lambda board, line_idx: board[line_idx*3:(line_idx+1)*3]
    visualize_line = lambda moves: "%s %s %s" % tuple(moves)

    board = self.__bit_to_array(state)
    visualized_moves = map(visualize_move, board)
    visualized_lines = [visualize_line(fetch_line(visualized_moves, i)) for i in range(3)]
    visualized_board = "\n".join(visualized_lines)
    return visualized_board


  @classmethod
  def __bit_to_array(self, state):
    board = [-1 for i in range(9)]
    for i in range(2):
      for j in range(9):
        if (state[i] >> j) & 1 == 1:
          board[j] = i
    return board[::-1]


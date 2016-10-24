from sample.ticktacktoe.ticktacktoe_domain import TickTackToeDomain

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

  @classmethod
  def measure_performance(self, domain, value_function, players):
    domains = [TickTackToeDomain(is_first_player=flg) for flg in [True, False]]
    next_is_first_player = lambda state: bin(state[0]|state[1]).count("1") % 2 == 0
    next_player_idx = lambda state: 0 if next_is_first_player(state) else 1
    state = domain.generate_initial_state()
    while not domain.is_terminal_state(state):
      idx = next_player_idx(state)
      player_domain, player = domains[idx], players[idx]
      action = player.choose_action(player_domain, value_function, state)
      state = domain.transit_state(state, action)
    return domain.calculate_reward(state)


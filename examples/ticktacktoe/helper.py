from kyoka.policy import BasePolicy, GreedyPolicy
from examples.ticktacktoe.task import TickTackToeTask

visualize_move = lambda flg: {-1: "-", 0:"O", 1:"X"}[flg]
fetch_line = lambda board, line_idx: board[line_idx*3:(line_idx+1)*3]
visualize_line = lambda moves: "%s %s %s" % tuple(moves)

def _bit_to_array(state):
    board = [-1 for i in range(9)]
    for i in range(2):
      for j in range(9):
        if (state[i] >> j) & 1 == 1:
          board[j] = i
    return board[::-1]


def visualize_board(state):
    board = _bit_to_array(state)
    visualized_moves = map(visualize_move, board)
    visualized_lines = [visualize_line(fetch_line(visualized_moves, i)) for i in range(3)]
    visualized_board = "\n".join(visualized_lines)
    return visualized_board

def measure_performance(task, value_function, players):
    tasks = [TickTackToeTask(is_first_player=flg) for flg in [True, False]]
    next_is_first_player = lambda state: bin(state[0]|state[1]).count("1") % 2 == 0
    next_player_idx = lambda state: 0 if next_is_first_player(state) else 1
    state = task.generate_initial_state()
    while not task.is_terminal_state(state):
        idx = next_player_idx(state)
        player_task, player = tasks[idx], players[idx]
        action = player.choose_action(player_task, value_function, state)
        state = task.transit_state(state, action)
    return task.calculate_reward(state)

def play_with_agent(value_func):
    next_is_first_player = lambda state: bin(state[0]|state[1]).count("1") % 2 == 0
    next_player_idx = lambda state: 0 if next_is_first_player(state) else 1
    def show_board(state):
        print "\n%s" % visualize_board(state)

    tasks = [TickTackToeTask(is_first_player=is_first) for is_first in [True, False]]
    players = [GreedyPolicy(), TickTackToeManualPolicy()]

    state = tasks[0].generate_initial_state()
    show_board(state)
    while not tasks[0].is_terminal_state(state):
        idx = next_player_idx(state)
        task, player = tasks[idx], players[idx]
        action = player.choose_action(task, value_func, state)
        state = task.transit_state(state, action)
        show_board(state)

def construct_features(task, state, action):
    next_state = task.transit_state(state, action)
    flg_to_ary = lambda flg: reduce(lambda acc, e: acc + [1 if (flg>>e)&1==1 else 0], range(9), [])
    multi_dim_ary = [flg_to_ary(player_board) for player_board in next_state]
    return multi_dim_ary[0] + multi_dim_ary[1]


class TickTackToeManualPolicy(BasePolicy):

    ACTION_NAME_MAP = {
            1 : "lower_right",
            2 : "lower_center",
            4 : "lower_left",
            8 : "middle_right",
            16: "middle_center",
            32: "middle_left",
            64: "upper_right",
            128: "upper_center",
            256: "upper_left"
    }

    def choose_action(self, task, value_function, state):
        message = self._ask_message(task, state) + " >> "
        action = int(raw_input(message))
        if action not in task.generate_possible_actions(state):
          return self.choose_action(task, value_function, state)
        return action

    def _ask_message(self, task, state):
        possible_actions = task.generate_possible_actions(state)
        names = [self.ACTION_NAME_MAP[action] for action in possible_actions]
        return ", ".join(["%d: %s" % info for info in zip(possible_actions, names)])


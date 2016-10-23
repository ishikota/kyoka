from kyoka.policy.greedy_policy import GreedyPolicy

class MazeHelper:

  @classmethod
  def visualize_maze(self, maze):
    return "\n".join(maze)

  @classmethod
  def measure_performance(self, domain, value_function, step_limit=10000):
    policy = GreedyPolicy()
    state = domain.generate_initial_state()
    step_counter = 0
    while not domain.is_terminal_state(state):
      action = policy.choose_action(domain, value_function, state)
      state = domain.transit_state(state, action)
      step_counter += 1
      if step_counter >= step_limit:
        break
    return step_counter

  @classmethod
  def visualize_policy(self, domain, value_function):
    icon_map = { domain.UP: "^", domain.DOWN: "v", domain.RIGHT: ">", domain.LEFT: "<", -1: "-", -2: "G" }
    actions = self.__find_best_actions_on_each_cell(domain ,value_function)
    flg2icon = lambda flg: icon_map[flg]
    visualized_actions = [[flg2icon(flg) for flg in line] for line in actions]
    return self.visualize_maze(["".join(line) for line in visualized_actions])


  @classmethod
  def __find_best_actions_on_each_cell(self, domain, value_function):
    height, width = domain.get_maze_shape()
    goal_r, goal_c = domain.generate_terminal_state()
    curry = lambda row, col: self.__find_single_best_action(domain, value_function, row, col)
    maze_with_answer = [[curry(row, col) for col in range(width)] for row in range(height)]
    maze_with_answer[goal_r][goal_c] = -2
    return maze_with_answer

  @classmethod
  def __find_single_best_action(self, domain, value_function, row, col):
    state = (row, col)
    actions = domain.generate_possible_actions(state)
    values = [value_function.calculate_value(state, action) for action in actions]
    best_actions = [act for act, val in zip(actions, values) if val == max(values)]
    if len(best_actions) == 1:
      return best_actions[0]
    else:
      return -1


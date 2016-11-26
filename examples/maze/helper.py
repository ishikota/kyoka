from kyoka.policy import GreedyPolicy

def visualize_maze(maze):
    return "\n".join(maze)

def measure_performance(task, value_function, step_limit=10000):
    policy = GreedyPolicy()
    state = task.generate_initial_state()
    step_counter = 0
    while not task.is_terminal_state(state):
        action = policy.choose_action(task, value_function, state)
        state = task.transit_state(state, action)
        step_counter += 1
        if step_counter >= step_limit: break
    return step_counter

def visualize_policy(task, value_function):
    icon_map = { task.UP: "^", task.DOWN: "v", task.RIGHT: ">", task.LEFT: "<", -1: "-", -2: "G" }
    actions = _find_best_actions_on_each_cell(task,value_function)
    flg2icon = lambda flg: icon_map[flg]
    visualized_actions = [[flg2icon(flg) for flg in line] for line in actions]
    return visualize_maze(["".join(line) for line in visualized_actions])

def construct_features(task, state, action):
    w, h = task.get_maze_shape()
    next_state = task.transit_state(state, action)
    onehot = [[1 if next_state == (row, col) else 0 for col in range(h)] for row in range(w)]
    return _flatten(onehot)

def _flatten(table):
    return [item for sublist in table for item in sublist]

def _find_best_actions_on_each_cell(task, value_function):
    height, width = task.get_maze_shape()
    goal_r, goal_c = task.generate_terminal_state()
    curry = lambda row, col: _find_single_best_action(task, value_function, row, col)
    maze_with_answer = [[curry(row, col) for col in range(width)] for row in range(height)]
    maze_with_answer[goal_r][goal_c] = -2
    return maze_with_answer

def _find_single_best_action(task, value_function, row, col):
    state = (row, col)
    actions = task.generate_possible_actions(state)
    values = [value_function.predict_value(state, action) for action in actions]
    best_actions = [act for act, val in zip(actions, values) if val == max(values)]
    if len(best_actions) == 1:
      return best_actions[0]
    else:
      return -1


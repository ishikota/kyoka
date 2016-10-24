from kyoka.domain.base_domain import BaseDomain
from kyoka.value_function.base_action_value_function import BaseActionValueFunction
from kyoka.policy.greedy_policy import GreedyPolicy
from kyoka.policy.epsilon_greedy_policy import EpsilonGreedyPolicy

from kyoka.algorithm.td_learning.sarsa import Sarsa
from kyoka.algorithm.td_learning.q_learning import QLearning
from kyoka.algorithm.td_learning.q_lambda import QLambda

class MazeDomain(BaseDomain):

  ACTION_UP = 0
  ACTION_DOWN = 1
  ACTION_RIGHT = 2
  ACTION_LEFT = 3

  # we use current position of the maze as "state". So here we return start position of the maze.
  def generate_initial_state(self):
    return (0, 0)

  # the position of the goal is (row=3, column=2)
  def is_terminal_state(self, state):
    return (0, 8) == state

  # we can always move to 4 directions.
  def generate_possible_actions(self, state):
    return [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_RIGHT, self.ACTION_LEFT]

  # RL algorithm can get reward only when he reaches to the goal.
  def calculate_reward(self, state):
    return 1 if self.is_terminal_state(state) else 0

  def transit_state(self, state, action):
    row, col = state
    wall_position = [(1,2), (2,2), (3,2), (4,5), (0,7), (1,7), (2,7)]
    height, width = 6, 9
    if action == self.ACTION_UP:
      row = max(0, row-1)
    elif action == self.ACTION_DOWN:
      row = min(height-1, row+1)
    elif action == self.ACTION_RIGHT:
      col= min(width-1, col+1)
    elif action == self.ACTION_LEFT:
      col = max(0, col-1)
    if (row, col) not in wall_position:
      return (row, col)
    else:
      return state # If destination is the wall or edge of the maze then position does not change.

class MazeActionValueFunction(BaseActionValueFunction):

  # call this method before start learning
  def setUp(self):
    maze_width, maze_height, action_num = 6, 9, 4
    self.table = [[[0 for k in range(action_num)] for j in range(maze_height)] for i in range(maze_width)]

  # just take value from the table
  def calculate_value(self, state, action):
    row, col = state
    return self.table[row][col][action]

  # just insert value into the table
  def update_function(self, state, action, new_value):
    row, col = state
    self.table[row][col][action] = new_value

class MazeHelper:

  @classmethod
  def visualize_maze(self, maze):
    return "\n".join(maze)

  @classmethod
  def visualize_policy(self, domain, value_function):
    icon_map = { domain.ACTION_UP: "^", domain.ACTION_DOWN: "v", domain.ACTION_RIGHT: ">", domain.ACTION_LEFT: "<", -1: "-", -2: "G" }
    actions = self.__find_best_actions_on_each_cell(domain ,value_function)
    flg2icon = lambda flg: icon_map[flg]
    visualized_actions = [[flg2icon(flg) for flg in line] for line in actions]
    return self.visualize_maze(["".join(line) for line in visualized_actions])


  @classmethod
  def __find_best_actions_on_each_cell(self, domain, value_function):
    height, width = 6, 9
    goal_r, goal_c = 0, 8
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

# You can replace RL algorithm like "rl_algo = Sarsa(alpha=0.1, gamma=0.7)"
#rl_algo = Sarsa(alpha=0.1, gamma=0.7)
rl_algo = QLearning(alpha=0.1, gamma=0.7)
#rl_algo = QLambda(alpha=0.1, gamma=0.7)
domain = MazeDomain()
value_function = MazeActionValueFunction()
policy = EpsilonGreedyPolicy(eps=0.1)
rl_algo.setUp(domain, policy, value_function)
rl_algo.run_gpi(nb_iteration=50)
print MazeHelper.visualize_policy(domain, value_function)

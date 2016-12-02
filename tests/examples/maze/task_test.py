import os

from nose.tools import raises

from examples.maze.task import MazeTask
from tests.base_unittest import BaseUnitTest


class MazeTaskTest(BaseUnitTest):

  def setUp(self):
    self.task = MazeTask()
    self.task.read_maze(get_sample_maze_path())

  def test_read_maze(self):
    self.eq('S', self.task.maze[0][1])
    self.eq('G', self.task.maze[1][3])
    self.eq('-', self.task.maze[0][0])
    self.eq('X', self.task.maze[1][2])

  def test_get_maze_shape(self):
    self.eq((3,4), self.task.get_maze_shape())

  def test_generate_inital_state(self):
    state = self.task.generate_initial_state()
    self.eq((0,1), state)

  @raises(ValueError)
  def test_raise_error_when_no_start_maze_passed(self):
    no_start_maze = [["-", "G"]]
    self.task.validate_maze(no_start_maze)

  @raises(ValueError)
  def test_raise_error_when_no_goal_maze_passed(self):
    no_goal_maze = [["-", "S"]]
    self.task.validate_maze(no_goal_maze)

  def test_terminal_state_check_on_start(self):
    start = (0,1)
    self.false(self.task.is_terminal_state(start))

  def test_terminal_state_check_on_goal(self):
    goal = (1, 3)
    self.true(self.task.is_terminal_state(goal))

  def test_transit_state_from_initial_state(self):
    transition_info = [
        (self.task.UP, (0, 1)),
        (self.task.LEFT, (0, 0)),
        (self.task.LEFT, (0, 0)),
        (self.task.DOWN, (1, 0)),
        (self.task.RIGHT, (1, 1)),
        (self.task.RIGHT, (1, 1)),
        (self.task.DOWN, (2, 1)),
        (self.task.DOWN, (2, 1)),
        (self.task.RIGHT, (2, 2)),
        (self.task.RIGHT, (2, 3)),
        (self.task.RIGHT, (2, 3)),
        (self.task.UP, (1, 3))
    ]
    state = (0,1)
    for action, expected in transition_info:
      state = self.__transit_and_check(state, action, expected)
    self.true(self.task.is_terminal_state(state))

  def test_generate_possible_actions(self):
    state = "dummy"
    expected = [self.task.UP, self.task.DOWN, self.task.RIGHT, self.task.LEFT]
    self.eq(expected, self.task.generate_possible_actions(state))

  def test_calculate_reward(self):
    start, goal, empty, block = (0, 1), (1, 3), (0, 0), (1, 2)
    self.eq(0, self.task.calculate_reward(start))
    self.eq(1, self.task.calculate_reward(goal))
    self.eq(0, self.task.calculate_reward(empty))
    self.eq(0, self.task.calculate_reward(block))


  def __transit_and_check(self, state, action, expected):
    next_state = self.task.transit_state(state, action)
    self.eq(expected, next_state)
    return next_state

def get_sample_maze_path():
    return os.path.join(os.path.dirname(__file__), "sample_maze.txt")


import os
from tests.base_unittest import BaseUnitTest
from kyoka.value_function.base_action_value_function import BaseActionValueFunction
from sample.maze.maze_domain import MazeDomain
from sample.maze.maze_helper import MazeHelper

class MazeHelperTest(BaseUnitTest):

  def setUp(self):
    self.domain = MazeDomain()
    self.domain.read_maze(self.__get_sample_maze_path())

  def test_visualize_maze(self):
    expected = \
        "-S--"+ "\n" + \
        "--XG"+ "\n" + \
        "----"
    self.eq(expected, MazeHelper.visualize_maze(self.domain.maze))

  def test_measure_performace(self):
    value_function = self.PerfectValueFunction()
    performance = MazeHelper.measure_performance(self.domain, value_function)
    self.eq(3, performance)

  def test_visualize_policy(self):
    value_function = self.PerfectValueFunction()
    expected = \
        "->>v"+ "\n" + \
        "---G"+ "\n" + \
        "----"
    self.eq(expected, MazeHelper.visualize_policy(self.domain, value_function))


  def test_find_best_actions_on_each_cell(self):
    value_function = self.PerfectValueFunction()
    answer = MazeHelper._MazeHelper__find_best_actions_on_each_cell(self.domain, value_function)
    self.eq(answer[0][0], -1)
    self.eq(answer[0][1], MazeDomain.RIGHT)
    self.eq(answer[0][2], MazeDomain.RIGHT)
    self.eq(answer[0][3], MazeDomain.DOWN)

  def __get_sample_maze_path(self):
    return os.path.join(os.path.dirname(__file__), "sample_maze.txt")

  class PerfectValueFunction(BaseActionValueFunction):

    CHEET_SHEET = {
        ((0,1), MazeDomain.RIGHT): 1,
        ((0,2), MazeDomain.RIGHT): 1,
        ((0,3), MazeDomain.DOWN): 1
    }

    def calculate_value(self, state, action):
      if (state, action) in self.CHEET_SHEET:
        return 1
      else:
        return 0


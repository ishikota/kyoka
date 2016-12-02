import os

import examples.maze.helper as H
from kyoka.value_function import BaseActionValueFunction
from examples.maze.task import MazeTask
from tests.base_unittest import BaseUnitTest


class MazeHelperTest(BaseUnitTest):

    def setUp(self):
        self.task = MazeTask()
        self.task.read_maze(get_sample_maze_path())

    def test_visualize_maze(self):
        expected = \
                "-S--"+ "\n" + \
                "--XG"+ "\n" + \
                "----"
        self.eq(expected, H.visualize_maze(self.task.maze))

    def test_measure_performace(self):
        value_function = PerfectValueFunction()
        performance = H.measure_performance(self.task, value_function)
        self.eq(3, performance)

    def test_visualize_policy(self):
        value_function = PerfectValueFunction()
        expected = \
                "->>v"+ "\n" + \
                "---G"+ "\n" + \
                "----"
        self.eq(expected, H.visualize_policy(self.task, value_function))

    def test_find_best_actions_on_each_cell(self):
        value_function = PerfectValueFunction()
        answer = H._find_best_actions_on_each_cell(self.task, value_function)
        self.eq(answer[0][0], -1)
        self.eq(answer[0][1], MazeTask.RIGHT)
        self.eq(answer[0][2], MazeTask.RIGHT)
        self.eq(answer[0][3], MazeTask.DOWN)

    def test_feature_construction(self):
        maze_shape = self.task.get_maze_shape()
        state1, action1 = (0, 0), self.task.RIGHT
        self.eq([0,1,0,0,0,0,0,0,0,0,0,0], H.construct_features(self.task, state1, action1))
        state2, action2 = (2, 1), self.task.UP
        self.eq([0,0,0,0,0,1,0,0,0,0,0,0], H.construct_features(self.task, state2, action2))
        state3, action3 = self.task.generate_terminal_state(), self.task.RIGHT
        self.eq([0,0,0,0,0,0,0,1,0,0,0,0], H.construct_features(self.task, state3, action3))

class PerfectValueFunction(BaseActionValueFunction):

    CHEET_SHEET = {
        ((0,1), MazeTask.RIGHT): 1,
        ((0,2), MazeTask.RIGHT): 1,
        ((0,3), MazeTask.DOWN): 1
    }

    def predict_value(self, state, action):
        if (state, action) in self.CHEET_SHEET:
            return 1
        else:
            return 0

def get_sample_maze_path():
    return os.path.join(os.path.dirname(__file__), "sample_maze.txt")


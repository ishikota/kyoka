import os
from tests.base_unittest import BaseUnitTest
from sample.maze.maze_dqn_value_function import MazeDQNValueFunction
from sample.maze.maze_domain import MazeDomain

class MazeDQNValueFunctionTest(BaseUnitTest):

  def setUp(self):
    self.domain = MazeDomain()
    self.domain.read_maze(self.__get_sample_maze_path())
    self.func = MazeDQNValueFunction(self.domain)
    self.func.setUp()

  def test_transform_state_action_into_input(self):
    test_func = lambda state, action: self.func._MazeDQNValueFunction__transform_state_action_into_input(state, action)
    state1, action1 = (0, 0), self.domain.RIGHT
    self.eq([0,1,0,0,0,0,0,0,0,0,0,0], test_func(state1, action1))
    state2, action2 = (2, 1), self.domain.UP
    self.eq([0,0,0,0,0,1,0,0,0,0,0,0], test_func(state2, action2))

  def xtest_prediction(self):
    state, action, target = (0, 0), self.domain.DOWN, 1
    learning_minibatch = [(state, action, target) for i in range(32)]
    self.func.train_on_minibatch(self.func.Q, learning_minibatch)
    trained_prediction = self.func.calculate_value(state, action)
    self.func.use_target_network(True)
    target_prediction = self.func.calculate_value(state, action)
    self.neq(trained_prediction, target_prediction)

  def xtest_deepcopy(self):
    state, action, target = (0, 0), self.domain.DOWN, 1
    learning_minibatch = [(state, action, target) for i in range(32)]
    self.func.train_on_minibatch(self.func.Q, learning_minibatch)
    self.func.reset_target_network()
    trained_prediction = self.func.calculate_value(state, action)
    self.func.use_target_network(True)
    target_prediction = self.func.calculate_value(state, action)
    self.almosteq(trained_prediction, target_prediction, 0.0001)


  def __get_sample_maze_path(self):
    return os.path.join(os.path.dirname(__file__), "sample_maze.txt")


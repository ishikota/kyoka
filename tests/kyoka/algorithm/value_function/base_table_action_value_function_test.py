from tests.base_unittest import BaseUnitTest
from kyoka.algorithm.value_function.base_table_action_value_function import BaseTableActionValueFunction

class BaseTableActionValueFunctionTest(BaseUnitTest):

  class TestImplementation(BaseTableActionValueFunction):

    def generate_initial_Q_table(self):
      return [[i+j for j in range(2)] for i in range(3)]

    def state_to_unique_value(self, state):
      return state % 3

    def action_to_unique_value(self, action):
      return action + 1


  def setUp(self):
    self.func = self.TestImplementation()
    self.func.setUp()

  def testUpdateAndCalculateStory(self):
    state, action = 5, 0  # Update Q_table[2][1]
    self.eq(3, self.func.calculate_action_value(state, action))
    self.func.update_function(state, action, new_value=10)
    self.eq(10, self.func.calculate_action_value(state, action))


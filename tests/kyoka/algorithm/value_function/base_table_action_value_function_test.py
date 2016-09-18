from tests.base_unittest import BaseUnitTest
from kyoka.algorithm.value_function.base_table_action_value_function import BaseTableActionValueFunction

class BaseTableActionValueFunctionTest(BaseUnitTest):

  def setUp(self):
    self.func = self.TestImplementation()
    self.func.setUp()

  def test_calculate_value(self):
    state, action = 0, 1
    self.func.table[state][action]= 1
    self.eq(1, self.func.calculate_value(state, action))

  def test_update_function(self):
    state, action = 0, 1
    self.eq(0, self.func.calculate_value(state, action))

    delta = self.func.update_function(state, action, 1)
    self.eq(1, self.func.calculate_value(state, action))
    self.eq(1, delta)

    delta = self.func.update_function(state, action, 0)
    self.eq(0, self.func.calculate_value(state, action))
    self.eq(-1, delta)

  class TestImplementation(BaseTableActionValueFunction):

    def generate_initial_table(self):
      return [[0 for j in range(2)] for i in range(1)]

    def fetch_value_from_table(self, table, state, action):
      return table[state][action]

    def update_table(self, table, state, action, new_value):
      table[state][action] = new_value
      return table


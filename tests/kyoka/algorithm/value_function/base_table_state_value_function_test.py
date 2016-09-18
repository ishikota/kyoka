from tests.base_unittest import BaseUnitTest
from kyoka.algorithm.value_function.base_table_state_value_function import BaseTableStateValueFunction

class BaseTableStateValueFunctionTest(BaseUnitTest):

  def setUp(self):
    self.func = self.TestImplementation()
    self.func.setUp()

  def test_calculate_value(self):
    state = 2
    self.func.table[state] = 1
    self.eq(1, self.func.calculate_value(state))

  def test_update_function(self):
    state = 2
    self.eq(0, self.func.calculate_value(state))

    delta = self.func.update_function(state, 1)
    self.eq(1, self.func.calculate_value(state))
    self.eq(1, delta)

    delta = self.func.update_function(state, 0)
    self.eq(0, self.func.calculate_value(state))
    self.eq(-1, delta)

  class TestImplementation(BaseTableStateValueFunction):

    def generate_initial_table(self):
      return [0 for i in range(3)]

    def fetch_value_from_table(self, table, state):
      return table[state]

    def update_table(self, table, state, new_value):
      table[state] = new_value
      return table


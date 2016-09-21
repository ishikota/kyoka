import os
from tests.base_unittest import BaseUnitTest
from kyoka.value_function.base_table_state_value_function import BaseTableStateValueFunction

class BaseTableStateValueFunctionTest(BaseUnitTest):

  def setUp(self):
    self.func = self.TestImplementation()
    self.func.setUp()

  def tearDown(self):
    file_path = self.__generate_tmp_file_path()
    if os.path.isfile(file_path):
      os.remove(file_path)

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

  def test_store_and_restore_table(self):
    file_path = self.__generate_tmp_file_path()
    state = 0
    self.func.update_function(state, 1)
    self.eq(1, self.func.calculate_value(state))
    self.func.save(file_path)
    self.func = self.TestImplementation()
    self.func.load(file_path)
    self.eq(1, self.func.calculate_value(state))


  def __generate_tmp_file_path(self):
    return os.path.join(os.path.dirname(__file__), "tmp_file_for_base_table_state_value_function_test.tmp")

  class TestImplementation(BaseTableStateValueFunction):

    def generate_initial_table(self):
      return [0 for i in range(3)]

    def fetch_value_from_table(self, table, state):
      return table[state]

    def update_table(self, table, state, new_value):
      table[state] = new_value
      return table


import os
from tests.base_unittest import BaseUnitTest
from kyoka.value_function.base_table_action_value_function import BaseTableActionValueFunction

class BaseTableActionValueFunctionTest(BaseUnitTest):

  def setUp(self):
    self.func = self.TestImplementation()
    self.func.setUp()

  def tearDown(self):
    file_path = self.__generate_tmp_file_path()
    if os.path.isfile(file_path):
      os.remove(file_path)

  def test_calculate_value(self):
    state, action = 0, 1
    self.func.table[state][action]= 1
    self.eq(1, self.func.calculate_value(state, action))

  def test_update_function(self):
    state, action = 0, 1
    self.eq(0, self.func.calculate_value(state, action))

    self.func.update_function(state, action, 1)
    self.eq(1, self.func.calculate_value(state, action))

    self.func.update_function(state, action, 0)
    self.eq(0, self.func.calculate_value(state, action))

  def test_store_and_restore_table(self):
    file_path = self.__generate_tmp_file_path()
    state, action = 0, 1
    self.func.update_function(state, action, 1)
    self.eq(1, self.func.calculate_value(state, action))
    self.func.save(file_path)
    self.func = self.TestImplementation()
    self.func.load(file_path)
    self.eq(1, self.func.calculate_value(state, action))


  def __generate_tmp_file_path(self):
    return os.path.join(os.path.dirname(__file__), "tmp_file_for_base_table_action_value_function_test.tmp")

  class TestImplementation(BaseTableActionValueFunction):

    def generate_initial_table(self):
      return [[0 for j in range(2)] for i in range(1)]

    def fetch_value_from_table(self, table, state, action):
      return table[state][action]

    def update_table(self, table, state, action, new_value):
      table[state][action] = new_value
      return table


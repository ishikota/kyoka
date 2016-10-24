import os
from tests.base_unittest import BaseUnitTest
from kyoka.value_function.base_table_state_value_function import BaseTableStateValueFunction

class BaseTableStateValueFunctionTest(BaseUnitTest):

  def setUp(self):
    self.func = self.TestImplementation()
    self.func.setUp()

  def tearDown(self):
    dir_path = self.__generate_tmp_dir_path()
    file_path = os.path.join(dir_path, "table_state_value_function_data.pickle")
    if os.path.exists(dir_path):
      if os.path.exists(file_path):
        os.remove(file_path)
      os.rmdir(dir_path)

  def test_calculate_value(self):
    state = 2
    self.func.table[state] = 1
    self.eq(1, self.func.calculate_value(state))

  def test_update_function(self):
    state = 2
    self.eq(0, self.func.calculate_value(state))

    self.func.update_function(state, 1)
    self.eq(1, self.func.calculate_value(state))

    self.func.update_function(state, 0)
    self.eq(0, self.func.calculate_value(state))

  def test_store_and_restore_table(self):
    dir_path = self.__generate_tmp_dir_path()
    file_path = os.path.join(dir_path, "table_state_value_function_data.pickle")
    os.mkdir(dir_path)
    state = 0
    self.func.update_function(state, 1)
    self.eq(1, self.func.calculate_value(state))
    self.false(os.path.exists(file_path))
    self.func.save(dir_path)
    self.true(os.path.exists(file_path))
    self.func = self.TestImplementation()
    self.func.load(dir_path)
    self.eq(1, self.func.calculate_value(state))

  def test_raise_error_when_load_failed(self):
    with self.assertRaises(IOError) as e:
      self.func.load("hoge")
    self.include("hoge", e.exception.message)


  def __generate_tmp_dir_path(self):
    return os.path.join(os.path.dirname(__file__), "tmp")

  class TestImplementation(BaseTableStateValueFunction):

    def generate_initial_table(self):
      return [0 for i in range(3)]

    def fetch_value_from_table(self, table, state):
      return table[state]

    def update_table(self, table, state, new_value):
      table[state] = new_value
      return table


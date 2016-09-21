import os
from tests.base_unittest import BaseUnitTest
from kyoka.value_function.base_value_function import BaseValueFunction
from kyoka.value_function.base_action_value_function import BaseActionValueFunction
from kyoka.value_function.base_state_value_function import BaseStateValueFunction

class BaseValueFunctionTest(BaseUnitTest):

  def tearDown(self):
    file_path = self.__generate_tmp_file_path()
    if os.path.isfile(file_path):
      os.remove(file_path)

  def test_deepcopy_default_implementation(self):
    func = BaseValueFunction()
    func.tmp = "hoge"
    copy = func.deepcopy()
    copy.tmp = "fuga"
    self.eq(func.tmp, copy.tmp)

  def test_save_load_function_with_action_value_function(self):
    file_path = self.__generate_tmp_file_path()
    original = self.TestActionValueFunction()
    original.state = "fuga"
    original.save(file_path)
    restored = self.TestActionValueFunction()
    self.eq("hoge", restored.state)
    restored.load(file_path)
    self.eq("fuga", restored.state)

  def test_save_load_function_with_state_value_function(self):
    file_path = self.__generate_tmp_file_path()
    original = self.TestStateValueFunction()
    original.state = "fuga"
    original.save(file_path)
    restored = self.TestStateValueFunction()
    self.eq("hoge", restored.state)
    restored.load(file_path)
    self.eq("fuga", restored.state)

  def test_save_load_additional_data(self):
    unique_key = "key4base_value_function_test"
    file_path = self.__generate_tmp_file_path()
    original = self.TestActionValueFunction()
    self.eq(None, original.get_additinal_data(unique_key))
    original.set_additinal_data(unique_key, "hoge")
    self.eq("hoge", original.get_additinal_data(unique_key))
    original.save(file_path)
    restored = self.TestActionValueFunction()
    self.eq(None, restored.get_additinal_data(unique_key))
    restored.load(file_path)
    self.eq("hoge", restored.get_additinal_data(unique_key))


  def __generate_tmp_file_path(self):
    return os.path.join(os.path.dirname(__file__), "tmp_file_for_base_value_function_test.tmp")

  class TestActionValueFunction(BaseActionValueFunction):

    def __init__(self):
      BaseActionValueFunction.__init__(self)
      self.state = "hoge"

    def provide_data_to_store(self):
      return self.state

    def receive_data_to_restore(self, restored_data):
      self.state = restored_data

  class TestStateValueFunction(BaseStateValueFunction):

    def __init__(self):
      BaseStateValueFunction.__init__(self)
      self.state = "hoge"

    def provide_data_to_store(self):
      return self.state

    def receive_data_to_restore(self, restored_data):
      self.state = restored_data


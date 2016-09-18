from tests.base_unittest import BaseUnitTest
from kyoka.algorithm.value_function.base_state_value_function import BaseStateValueFunction

class BaseStateValueFunctionTest(BaseUnitTest):

  def setUp(self):
    self.func = BaseStateValueFunction()

  def test_error_msg_when_not_implement_abstract_method(self):
    try:
      self.func.calculate_value("dummy")
    except NotImplementedError as e:
      self.include("BaseStateValueFunction", str(e))
      self.include("calculate_value", str(e))
    else:
      self.fail("NotImplementedError does not occur")


from tests.base_unittest import BaseUnitTest
from kyoka.algorithm.value_function.base_action_value_function import BaseActionValueFunction

class BaseActionValueFunctionTest(BaseUnitTest):

  def setUp(self):
    self.func = BaseActionValueFunction()

  def test_error_msg_when_not_implement_abstract_method(self):
    try:
      self.func.calculate_value("dummy", "dummy")
    except NotImplementedError as e:
      self.include("BaseActionValueFunction", str(e))
      self.include("calculate_value", str(e))
    else:
      self.fail("NotImplementedError does not occur")


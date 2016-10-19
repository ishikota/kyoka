from tests.base_unittest import BaseUnitTest
from kyoka.algorithm.td_learning.base_td_method import BaseTDMethod
from kyoka.value_function.base_state_value_function import BaseStateValueFunction

from mock import Mock
from nose.tools import raises

class BaseTDMethodTest(BaseUnitTest):

  def setUp(self):
    self.algo = BaseTDMethod()

  @raises(TypeError)
  def test_reject_state_value_function(self):
    value_func = Mock(spec=BaseStateValueFunction)
    self.algo.update_value_function("dummy", "dummy", value_func)


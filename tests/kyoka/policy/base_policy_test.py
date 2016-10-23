from tests.base_unittest import BaseUnitTest
from kyoka.domain.base_domain import BaseDomain
from kyoka.policy.base_policy import BasePolicy
from kyoka.value_function.base_action_value_function import BaseActionValueFunction
from kyoka.value_function.base_state_value_function import BaseStateValueFunction

from nose.tools import raises

class BasePolicyTest(BaseUnitTest):

  def setUp(self):
    self.domain = BaseDomain()
    self.value_func = BaseActionValueFunction()
    self.policy = BasePolicy()

  def test_error_msg_when_not_implement_abstract_method(self):
    try:
      self.policy.choose_action(None, None, None, None)
    except NotImplementedError as e:
      self.include("BasePolicy", str(e))
    else:
      self.fail("NotImplementedError does not occur")

  def test_pack_arguments_for_value_function_when_action_value_function(self):
    value_func = BaseActionValueFunction()
    packed = self.policy.pack_arguments_for_value_function(value_func, "state", "action")
    self.eq(["state", "action"], packed)

  def test_pack_arguments_for_value_function_when_state_value_function(self):
    value_func = BaseStateValueFunction()
    packed = self.policy.pack_arguments_for_value_function(value_func, "state", "action")
    self.eq(["state"], packed)

  @raises(ValueError)
  def test_pack_arguments_for_value_function_when_illegal_state(self):
    self.policy.pack_arguments_for_value_function("dummy", "state", "action")


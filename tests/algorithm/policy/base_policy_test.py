from tests.base_unittest import BaseUnitTest
from kyoka.domain.base_domain import BaseDomain
from kyoka.algorithm.policy.base_policy import BasePolicy

class BasePolicyTest(BaseUnitTest):

  def setUp(self):
    domain = BaseDomain()
    self.policy = BasePolicy(domain)

  def test_error_msg_when_not_implement_abstract_method(self):
    try:
      self.policy.choose_action(None, None)
    except NotImplementedError as e:
      self.include("BasePolicy", str(e))
    else:
      self.fail("NotImplementedError does not occur")


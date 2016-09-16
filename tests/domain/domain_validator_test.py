from tests.base_unittest import BaseUnitTest
from kyoka.domain.base_domain import BaseDomain
from kyoka.domain.domain_validator import DomainValidator

class DomainValidatorTest(BaseUnitTest):

  def test_implementation_check(self):
    empty_domain = BaseDomain()
    validator = DomainValidator(empty_domain)
    status, msg = validator.implementation_check()
    self.false(status)
    self.include("generate_inital_state", msg)


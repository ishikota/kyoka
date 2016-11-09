from tests.base_unittest import BaseUnitTest
from kyoka.value_function_ import BaseTabularActionValueFunction, BaseApproxActionValueFunction
from nose.tools import raises

class BaseTabularActionValueFunctionTest(BaseUnitTest):

    def setUp(self):
        self.func = BaseTabularActionValueFunction()

    @raises(NotImplementedError)
    def test_error_msg_when_not_implement_predict_value(self):
        self.func.predict_value("dummy", "dummy")

    @raises(NotImplementedError)
    def test_error_msg_when_not_implement_backup(self):
        self.func.backup("dummy", "dummy", "dummy", "dummy")

class BaseApproxActionValueFunctionTest(BaseUnitTest):

    def setUp(self):
        self.func = BaseApproxActionValueFunction()

    @raises(NotImplementedError)
    def test_error_msg_when_not_implement_calculate_value(self):
        self.func.predict_value("dummy")

    @raises(NotImplementedError)
    def test_error_msg_when_not_implement_update_function(self):
        self.func.backup("dummy", "dummy", "dummy")


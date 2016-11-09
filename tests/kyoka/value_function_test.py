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

    def test_predoct_value(self):
        func = self.TestImpl()
        self.eq("predict:hogefuga", func.predict_value("hoge", "fuga"))

    def test_backup(self):
        func = self.TestImpl()
        func.backup("hoge", "fuga", "dummy", "dummy")
        self.eq("backup:hogefuga", func.memo)

    @raises(NotImplementedError)
    def test_error_msg_when_not_implement_construct_features(self):
        self.func.construct_features("dummy", "dummy")

    @raises(NotImplementedError)
    def test_error_msg_when_not_implement_approx_predict_value(self):
        self.func.approx_predict_value("dummy")

    @raises(NotImplementedError)
    def test_error_msg_when_not_implement_approx_backup(self):
        self.func.approx_backup("dummy", "dummy", "dummy")

    class TestImpl(BaseApproxActionValueFunction):

        def construct_features(self, state, action):
            return state + action

        def approx_predict_value(self, features):
            return "predict:%s" % features

        def approx_backup(self, features, backup_target, alpha):
            self.memo = "backup:%s" % features


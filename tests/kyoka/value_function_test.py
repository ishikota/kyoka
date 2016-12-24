import os

from nose.tools import raises

from kyoka.value_function import BaseTabularActionValueFunction, BaseApproxActionValueFunction
from tests.base_unittest import BaseUnitTest
from tests.utils import generate_tmp_dir_path, setup_tmp_dir, teardown_tmp_dir


class BaseTabularActionValueFunctionTest(BaseUnitTest):

    def setUp(self):
        self.func = self.TestImpl()
        self.func.setup()

    def tearDown(self):
        tmp_file_name = "hoge_table_action_value_function_data.pickle"
        teardown_tmp_dir(__file__, [tmp_file_name])

    @raises(NotImplementedError)
    def test_backup(self):
        BaseTabularActionValueFunction().backup("dummy", "dummy", "dummy", "dummy")

    @raises(NotImplementedError)
    def test_generate_initial_table(self):
        BaseTabularActionValueFunction().generate_initial_table()

    @raises(NotImplementedError)
    def test_fetch_value_from_table(self):
        BaseTabularActionValueFunction().fetch_value_from_table("dummy", "dummy", "dummy")

    @raises(NotImplementedError)
    def test_insert_value_into_table(self):
        BaseTabularActionValueFunction().insert_value_into_table("dummy", "dummy", "dummy", "dummy")

    def test_predict_value(self):
        state, action = 0, 1
        self.func.table[state][action]= 1
        self.eq(1, self.func.predict_value(state, action))

    def test_insert_value_into_table(self):
        state, action = 0, 1
        self.eq(0, self.func.predict_value(state, action))
        self.func.insert_value_into_table(self.func.table, state, action, 1)
        self.eq(1, self.func.predict_value(state, action))

    def test_store_and_restore_table(self):
        setup_tmp_dir(__file__)
        dir_path = generate_tmp_dir_path(__file__)
        file_path = os.path.join(dir_path, "hoge_table_action_value_function_data.pickle")
        state, action = 0, 1
        self.func.insert_value_into_table(self.func.table, state, action, 1)
        self.eq(1, self.func.predict_value(state, action))
        self.false(os.path.exists(file_path))
        self.func.save(dir_path)
        self.true(os.path.exists(file_path))
        self.func = self.TestImpl()
        self.func.load(dir_path)
        self.eq(1, self.func.predict_value(state, action))

    class TestImpl(BaseTabularActionValueFunction):

        def generate_initial_table(self):
            return [[0 for j in range(2)] for i in range(1)]

        def fetch_value_from_table(self, table, state, action):
            return table[state][action]

        def insert_value_into_table(self, table, state, action, new_value):
            table[state][action] = new_value

        def define_save_file_prefix(self):
            return "hoge"


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


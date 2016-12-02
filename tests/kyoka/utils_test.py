import kyoka.utils as U
from tests.base_unittest import BaseUnitTest


class UtilsTest(BaseUnitTest):

    def test_build_not_implemented_msg(self):
        message = U.build_not_implemented_msg(self, "hoge")
        self.include("UtilsTest", message)
        self.include("hoge", message)

    def test_value_function_check(self):
        with self.assertRaises(TypeError) as e:
            U.value_function_check("hoge", [BaseUnitTest, UtilsTest], "value_function")
        self.include("hoge", e.exception.message)
        self.include("BaseUnitTest or UtilsTest", e.exception.message)


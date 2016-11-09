from tests.base_unittest import BaseUnitTest
import kyoka.utils as U

class UtilsTest(BaseUnitTest):

    def test_build_not_implemented_msg(self):
        message = U.build_not_implemented_msg(self, "hoge")
        self.include("UtilsTest", message)
        self.include("hoge", message)


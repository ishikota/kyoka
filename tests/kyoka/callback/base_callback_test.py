from tests.base_unittest import BaseUnitTest
from kyoka.callback.base_callback import BaseCallback

import sys
import StringIO

class BaseCallbackTest(BaseUnitTest):

  def setUp(self):
    self.callback = BaseCallback()
    self.capture = StringIO.StringIO()
    sys.stdout = self.capture

  def tearDown(self):
    sys.stdout = sys.__stdout__

  def test_interrupt_gpi(self):
    self.false(self.callback.interrupt_gpi("dummy", "dummy", "dummy"))

  def test_define_log_tag(self):
    self.eq("BaseCallback", self.callback.define_log_tag())

  def test_log(self):
    self.callback.log("hoge")
    self.eq("[BaseCallback] hoge\n", self.capture.getvalue())


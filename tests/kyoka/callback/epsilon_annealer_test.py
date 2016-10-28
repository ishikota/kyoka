from tests.base_unittest import BaseUnitTest
from kyoka.callback.epsilon_annealer import EpsilonAnnealer
from kyoka.policy.epsilon_greedy_policy import EpsilonGreedyPolicy
import sys
import StringIO

class EpsilonAnnealerTest(BaseUnitTest):

  def setUp(self):
    self.policy = EpsilonGreedyPolicy()
    self.policy.set_eps_annealing(1.0, 0.1, 100)
    self.annealer = EpsilonAnnealer(self.policy)
    self.capture = StringIO.StringIO()
    sys.stdout = self.capture

  def tearDown(self):
    sys.stdout = sys.__stdout__

  def test_define_log_tag(self):
    self.eq("EpsilonGreedyAnnealing", self.annealer.define_log_tag())

  def test_start_log(self):
    self.annealer.before_gpi_start("dummy", "dummy")
    self.include(str(1.0), self.capture.getvalue())
    self.include(str(0.1), self.capture.getvalue())

  def test_finish_log(self):
    [self.annealer.after_update(_, "dummy", "dummy") for _ in range(99)]
    self.not_include("finish", self.capture)
    self.annealer.after_update(100, "dummy", "dummy")
    self.include("finish", self.capture.getvalue())
    self.include(str(100), self.capture.getvalue())
    # Check that finish message is logged only once
    capture = StringIO.StringIO()
    sys.stdout = capture
    self.annealer.after_update(101, "dummy", "dummy")
    self.not_include("finish", capture.getvalue())


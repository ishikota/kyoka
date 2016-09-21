from collections import defaultdict
from tests.base_unittest import BaseUnitTest
from kyoka.algorithm.td_learning.eligibility_trace.action_eligibility_trace import ActionEligibilityTrace

from nose.tools import raises

class ActionEligibilityTraceTest(BaseUnitTest):

  def test_default_value(self):
    trace = ActionEligibilityTrace("accumulating_trace", gamma=0.5, lambda_=0.1)
    self.eq(0, trace.get(0, 0))
    self.eq(0, trace.get(100, 200))

  def test_accumulating_update(self):
    trace = ActionEligibilityTrace("accumulating_trace", gamma=0.5, lambda_=0.1)
    trace.update(100, 200)
    self.eq(1, trace.get(100, 200))
    trace.update(100, 200)
    self.eq(2, trace.get(100, 200))

  def test_replacing_update(self):
    trace = ActionEligibilityTrace("replacing_trace", gamma=0.5, lambda_=0.1)
    trace.update(100, 200)
    self.eq(1, trace.get(100, 200))
    trace.update(100, 200)
    self.eq(1, trace.get(100, 200))

  def test_decay(self):
    trace = ActionEligibilityTrace("accumulating_trace", gamma=0.5, lambda_=0.1)
    trace.update(100, 200)
    trace.decay(100, 200)
    self.eq(0.05, trace.get(100, 200))

  def test_decay_discard(self):
    trace = ActionEligibilityTrace(\
        "accumulating_trace", discard_threshold=0.0001, gamma=0.99, lambda_=0.9)
    trace.update(0, 0)
    trace.update(100, 200)
    [trace.decay(100, 200) for _ in range(79)]
    self.neq(0, trace.get(100, 200))
    trace.decay(100, 200)
    self.eq([(0, 0, 1)], trace.get_eligibilities())

  def test_get_eligibilities(self):
    trace = ActionEligibilityTrace("accumulating_trace", gamma=0.5, lambda_=0.1)
    trace.update(0, 0)
    trace.update(0, 0)
    trace.update(100, 200)
    expected = { 0 : { 0 : 2 }, 100: { 200: 1 } }
    eligibilities = trace.get_eligibilities()
    self.eq(2, len(eligibilities))
    for state, action, eligibility in eligibilities:
      self.eq(expected[state][action], eligibility)

  def test_clear(self):
    trace = ActionEligibilityTrace("accumulating_trace", gamma=0.5, lambda_=0.1)
    trace.update(0, 0)
    trace.update(100, 200)
    trace.clear()
    self.eq(0, trace.get(0, 0))
    self.eq(0, trace.get(100, 2000))

  @raises(TypeError)
  def test_update_type_validation_when_initialize(self):
    ActionEligibilityTrace("invalid_type", gamma=0.5, lambda_=0.1)

  @raises(TypeError)
  def test_update_type_validation_when_update(self):
    trace = ActionEligibilityTrace("accumulating_trace", gamma=0.5, lambda_=0.1)
    trace.update_type = "invalid_type"
    trace.update(0, 0)

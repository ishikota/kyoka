from nose.tools import raises

from kyoka.task import BaseTask
from tests.base_unittest import BaseUnitTest


class BaseTaskTest(BaseUnitTest):

    def setUp(self):
        self.task = BaseTask()

    @raises(NotImplementedError)
    def test_generate_inital_state(self):
        self.task.generate_inital_state()

    @raises(NotImplementedError)
    def test_is_terminal_state(self):
        self.task.is_terminal_state("dummy")

    @raises(NotImplementedError)
    def test_transit_state(self):
        self.task.transit_state("dummy", "dummy")

    @raises(NotImplementedError)
    def test_generate_possible_actions(self):
        self.task.generate_possible_actions("dummy")

    @raises(NotImplementedError)
    def test_calculate_reward(self):
        self.task.calculate_reward("dummy")


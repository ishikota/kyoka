from tests.base_unittest import BaseUnitTest
from kyoka.algorithm.base_rl_algorithm import BaseRLAlgorithm
from kyoka.policy.greedy_policy import GreedyPolicy
from kyoka.policy.epsilon_greedy_policy import EpsilonGreedyPolicy
from kyoka.value_function.base_action_value_function import BaseActionValueFunction
from kyoka.callback.finish_rule.base_finish_rule import BaseFinishRule

from mock import Mock

import StringIO
import sys

class BaseRLAlgorithmTest(BaseUnitTest):

  def setUp(self):
    self.algo = BaseRLAlgorithm()
    self.capture = StringIO.StringIO()
    sys.stdout = self.capture

  def tearDown(self):
    sys.stdout = sys.__stdout__

  def test_setUp(self):
    algo = BaseRLAlgorithm()
    mock_func = Mock()
    algo.setUp(domain=0, policy=1, value_function=mock_func)
    self.eq(0, algo.domain)
    self.eq(1, algo.policy)
    self.eq(mock_func, algo.value_function)
    mock_func.setUp.assert_called()

  def test_save(self):
    algo = self.TestImplementation()
    mock_func = Mock()
    algo.setUp(domain=0, policy=1, value_function=mock_func)
    algo.save("hoge")
    mock_func.save.assert_called_with("hoge")
    self.eq(1, algo.save_count)
    self.eq(0, algo.load_count)

  def test_load(self):
    algo = self.TestImplementation()
    mock_func = Mock()
    algo.setUp(domain=0, policy=1, value_function=mock_func)
    algo.load("hoge")
    mock_func.load.assert_called_with("hoge")
    self.eq(0, algo.save_count)
    self.eq(1, algo.load_count)


  def test_error_msg_when_not_implement_abstract_method(self):
    self.__check_err_msg(lambda : self.algo.update_value_function("dummy", "dummy", "dummy"), "update_value_function")

  def test_gen_episode(self):
    domain = self.__setup_stub_domain()
    policy = GreedyPolicy()
    value_func = self.__setup_stub_value_function()
    self.algo.setUp(domain, policy, value_func)
    episode = self.algo.generate_episode(domain, value_func, policy)
    self.eq(3, len(episode))
    self.eq((0, 1, 1, 1), episode[0])
    self.eq((1, 2, 3, 9), episode[1])
    self.eq((3, 4, 7, 49), episode[2])

  def test_GPI(self):
    algo = self.TestImplementation()
    domain = self.__setup_stub_domain()
    policy = GreedyPolicy()
    value_func = self.__setup_stub_value_function()
    algo.setUp(domain, policy, value_func)
    finish_rule = self.TestFinishRule()
    algo.run_gpi(nb_iteration=3, callbacks=finish_rule)
    expected = "[test_tag] finish:2\n"
    self.include(expected, self.capture.getvalue())

  def test_GPI_with_multiple_finish_rules(self):
    algo = self.TestImplementation()
    domain = self.__setup_stub_domain()
    policy = GreedyPolicy()
    value_func = self.__setup_stub_value_function()
    algo.setUp(domain, policy, value_func)
    finish_rule1 = self.TestFinishRule([False, True])
    finish_rule2 = self.TestFinishRule([True, False])
    finish_rules = [finish_rule1, finish_rule2]
    finish_msg = algo.run_gpi(nb_iteration=2, callbacks=finish_rules)
    expected = 1
    expected = "[test_tag] finish:1\n"
    self.include(expected, self.capture.getvalue())

  def test_verbose_mode(self):
    algo = self.TestImplementation()
    domain = self.__setup_stub_domain()
    policy = GreedyPolicy()
    value_func = self.__setup_stub_value_function()
    algo.setUp(domain, policy, value_func)

    capture = StringIO.StringIO()
    sys.stdout = capture
    algo.run_gpi(nb_iteration=2, callbacks=self.TestFinishRule())
    self.include("[Progress] Finished", capture.getvalue())

    capture = StringIO.StringIO()
    sys.stdout = capture
    algo.run_gpi(nb_iteration=2, callbacks=self.TestFinishRule(), verbose=0)
    self.not_include("[Progress] Finished", capture.getvalue())

  def test_defult_finish_message_must_be_logged(self):
    algo = self.TestImplementation()
    domain = self.__setup_stub_domain()
    policy = GreedyPolicy()
    value_func = self.__setup_stub_value_function()
    algo.setUp(domain, policy, value_func)
    finish_rule = self.TestFinishRule()
    algo.run_gpi(nb_iteration=2, callbacks=finish_rule)

    expected = "[Progress] Completed"
    self.include(expected, self.capture.getvalue())

  def test_not_to_log_duplicated_default_finish_message(self):
    algo = self.TestImplementation()
    domain = self.__setup_stub_domain()
    policy = GreedyPolicy()
    value_func = self.__setup_stub_value_function()
    algo.setUp(domain, policy, value_func)
    algo.run_gpi(nb_iteration=2)
    self.eq(1, self.capture.getvalue().count("[Progress] Completed"))

  def test_default_annealing_message(self):
    algo = self.TestImplementation()
    domain = self.__setup_stub_domain()
    policy = EpsilonGreedyPolicy()
    value_func = self.__setup_stub_value_function()

    algo.setUp(domain, policy, value_func)
    algo.run_gpi(nb_iteration=2)
    self.not_include("[EpsilonGreedyAnnealing]", self.capture.getvalue())

    policy.set_eps_annealing(1.0, 0.1, 100)
    algo.setUp(domain, policy, value_func)
    algo.run_gpi(nb_iteration=2)
    self.include("[EpsilonGreedyAnnealing]", self.capture.getvalue())

  def test_set_callback(self):
    algo = self.TestImplementation()
    value_func = Mock(name="value_func")
    algo.setUp("domain", "dummy", value_func)
    callback = Mock()
    callback.interrupt_gpi.return_value = False
    finish_rule = self.TestFinishRule()
    finish_msg = algo.run_gpi(nb_iteration=2, callbacks=[callback, finish_rule])
    self.eq(1, callback.before_gpi_start.call_count)
    self.eq(2, callback.before_update.call_count)
    self.eq(2, callback.after_update.call_count)
    self.eq(1, callback.after_gpi_finish.call_count)
    callback.before_gpi_start.assert_called_with("domain", value_func)
    callback.before_update.assert_called_with(2, "domain", value_func)
    callback.after_update.assert_called_with(2, "domain", value_func)
    callback.after_gpi_finish.assert_called_with("domain", value_func)

  def test_error_when_run_gpi_called_without_setup(self):
    algo = self.TestImplementation()
    with self.assertRaises(Exception) as e:
      algo.run_gpi(nb_iteration=2)
    self.include("setUp", e.exception.message)
    self.include("run_gpi", e.exception.message)


  def __setup_stub_domain(self):
    mock_domain = Mock()
    mock_domain.generate_initial_state.return_value = 0
    mock_domain.is_terminal_state.side_effect = lambda state: state == 7
    mock_domain.transit_state.side_effect = lambda state, action: state + action
    mock_domain.generate_possible_actions.side_effect = lambda state: [state + 1]
    mock_domain.calculate_reward.side_effect = lambda state: state**2
    return mock_domain

  def __setup_stub_value_function(self):
    mock_value_func = Mock(spec=BaseActionValueFunction)
    mock_value_func.calculate_value.return_value = 0
    return mock_value_func

  def __check_err_msg(self, target_method, target_word):
    try:
      target_method()
    except NotImplementedError as e:
      self.include(target_word, str(e))
    else:
      self.fail("NotImplementedError does not occur")

  class TestFinishRule(BaseFinishRule):

    def __init__(self, return_value=[False, True]):
      BaseFinishRule.__init__(self)
      self.return_value = return_value
      self.return_idx = 0

    def define_log_tag(self):
      return "test_tag"

    def check_condition(self, iteration_count, _domain, _value_function):
      self.return_idx += 1
      return self.return_value[self.return_idx-1]

    def generate_start_message(self):
      return ""

    def generate_finish_message(self, iteration_count):
      return "%s:%s" % ("finish", iteration_count)


  class TestImplementation(BaseRLAlgorithm):

    def __init__(self):
      BaseRLAlgorithm.__init__(self)
      self.update_count = 0
      self.save_count = 0
      self.load_count = 0

    def save_algorithm_state(self, save_dir_path):
      self.save_count += 1

    def load_algorithm_state(self, load_dir_path):
      self.load_count += 1

    def update_value_function(self, _domain, _policy, _value_function):
      self.update_count += 1
      return self.update_count


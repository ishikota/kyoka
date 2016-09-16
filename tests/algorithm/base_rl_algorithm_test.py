from tests.base_unittest import BaseUnitTest
from kyoka.algorithm.base_rl_algorithm import BaseRLAlgorithm

class BasePolicyTest(BaseUnitTest):

  def setUp(self):
    self.algo = BaseRLAlgorithm()

  def test_error_msg_when_not_implement_abstract_method(self):
    self.__check_err_msg(lambda : self.algo.training("dummy", "dummy"), "training")
    self.__check_err_msg(lambda : self.algo.select_action("dummy", "dummy"), "select_action")
    self.__check_err_msg(lambda : self.algo.save_value_function("dummy"), "save_value_function")
    self.__check_err_msg(lambda : self.algo.load_value_function("dummy"), "load_value_function")


  def __check_err_msg(self, target_method, target_word):
    try:
      target_method()
    except NotImplementedError as e:
      self.include(target_word, str(e))
    else:
      self.fail("NotImplementedError does not occur")


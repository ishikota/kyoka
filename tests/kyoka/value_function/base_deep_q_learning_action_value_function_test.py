from tests.base_unittest import BaseUnitTest
from kyoka.value_function.base_deep_q_learning_action_value_function import BaseDeepQLearningActionValueFunction
from nose.tools import raises

class BaseDeepQLearningActionValueFunctionTest(BaseUnitTest):

  def setUp(self):
    self.func = BaseDeepQLearningActionValueFunction()

  def test_initialize_network(self):
    with self.assertRaises(NotImplementedError) as e:
      self.func.initialize_network()
    self.include("initialize_network", e.exception.message)

  def test_deepcopy_network(self):
    with self.assertRaises(NotImplementedError) as e:
      self.func.deepcopy_network("dummy")
    self.include("deepcopy_network", e.exception.message)

  def test_preprocess_state(self):
    with self.assertRaises(NotImplementedError) as e:
      self.func.preprocess_state("dummy")
    self.include("preprocess_state", e.exception.message)

  def test_predict_action_value(self):
    with self.assertRaises(NotImplementedError) as e:
      self.func.predict_action_value("dummy", "dummy", "dummy")
    self.include("predict_action_value", e.exception.message)

  def test_train_on_minibatch(self):
    with self.assertRaises(NotImplementedError) as e:
      self.func.train_on_minibatch("dummy", "dummy")
    self.include("train_on_minibatch", e.exception.message)

  def test_save_model_weights(self):
    with self.assertRaises(NotImplementedError) as e:
      self.func.save_model_weights("dummy")
    self.include("save_model_weights", e.exception.message)

  def test_load_model_weights(self):
    with self.assertRaises(NotImplementedError) as e:
      self.func.load_model_weights("dummy")
    self.include("load_model_weights", e.exception.message)

  def test_update_function(self):
    with self.assertRaises(AttributeError) as e:
      self.func.update_function("dummy", "dummy", "dummy")
    self.include("update_function", e.exception.message)

  def test_setup(self):
    func = self.TestImple()
    func.setUp()
    self.eq(0, func.Q)
    self.eq(1, func.Q_hat)
    self.false(func.use_target_network_flg)

  def test_use_target_network(self):
    func = self.TestImple()
    func.setUp()
    func.use_target_network(True)
    self.true(func.use_target_network_flg)
    func.use_target_network(False)
    self.false(func.use_target_network_flg)

  def test_calculate_value(self):
    func = self.TestImple()
    func.setUp()
    self.eq(0, func.calculate_value(2, 3))
    func.use_target_network(True)
    self.eq(6, func.calculate_value(2, 3))

  def test_reset_target_network(self):
    func = self.TestImple()
    func.setUp()
    func.Q = 2
    func.reset_target_network()
    self.eq(3, func.Q_hat)


  class TestImple(BaseDeepQLearningActionValueFunction):

    def initialize_network(self):
      return 0

    def deepcopy_network(self, q_network):
      return q_network + 1

    def predict_action_value(self, network, state, action):
      return network * state * action


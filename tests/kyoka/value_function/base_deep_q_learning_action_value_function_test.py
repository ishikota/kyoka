from tests.base_unittest import BaseUnitTest
from kyoka.value_function.base_deep_q_learning_action_value_function import BaseDeepQLearningActionValueFunction
from nose.tools import raises

import pickle
import os

class BaseDeepQLearningActionValueFunctionTest(BaseUnitTest):

  def setUp(self):
    self.func = BaseDeepQLearningActionValueFunction()

  def tearDown(self):
    dir_path = self.__generate_tmp_dir_path()
    file_path = os.path.join(dir_path, "dqn_value_function_data.pickle")
    if os.path.exists(dir_path):
      if os.path.exists(file_path):
        os.remove(file_path)
      os.rmdir(dir_path)

  def test_initialize_network(self):
    with self.assertRaises(NotImplementedError) as e:
      self.func.initialize_network()
    self.include("initialize_network", e.exception.message)

  def test_deepcopy_network(self):
    with self.assertRaises(NotImplementedError) as e:
      self.func.deepcopy_network("dummy")
    self.include("deepcopy_network", e.exception.message)

  def test_preprocess_state_sequence(self):
    with self.assertRaises(NotImplementedError) as e:
      self.func.preprocess_state_sequence("dummy")
    self.include("preprocess_state_sequence", e.exception.message)

  def test_predict_action_value(self):
    with self.assertRaises(NotImplementedError) as e:
      self.func.predict_action_value("dummy", "dummy", "dummy")
    self.include("predict_action_value", e.exception.message)

  def test_train_on_minibatch(self):
    with self.assertRaises(NotImplementedError) as e:
      self.func.train_on_minibatch("dummy", "dummy")
    self.include("train_on_minibatch", e.exception.message)

  def test_save_networks(self):
    with self.assertRaises(NotImplementedError) as e:
      self.func.save_networks("dummy", "dummy", "dummy")
    self.include("save_networks", e.exception.message)

  def test_load_networks(self):
    with self.assertRaises(NotImplementedError) as e:
      self.func.load_networks("dummy")
    self.include("load_networks", e.exception.message)

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

  def test_save_and_load_networks(self):
    func = self.TestImple()
    func.setUp()
    func.Q = 2
    dir_path = self.__generate_tmp_dir_path()
    os.mkdir(dir_path)
    func.save(dir_path)
    new_func = self.TestImple()
    new_func.setUp()
    new_func.load(dir_path)
    self.eq(func.Q, new_func.Q)
    self.eq(func.Q_hat, new_func.Q_hat)

  def __generate_tmp_dir_path(self):
    return os.path.join(os.path.dirname(__file__), "tmp")

  class TestImple(BaseDeepQLearningActionValueFunction):

    def initialize_network(self):
      return 0

    def deepcopy_network(self, q_network):
      return q_network + 1

    def predict_action_value(self, network, state, action):
      return network * state * action

    def save_networks(self, Q_network, Q_hat_network, save_dir_path):
      with open(self.generate_tmp_file_path(), "wb") as f:
        pickle.dump((Q_network, Q_hat_network), f)

    def load_networks(self, load_dir_path):
      with open(self.generate_tmp_file_path(), "rb") as f:
        Q_network, Q_hat_network = pickle.load(f)
      return Q_network, Q_hat_network

    def generate_tmp_file_path(self):
      return os.path.join(os.path.dirname(__file__), "dqn_value_function_data.pickle")


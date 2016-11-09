from kyoka.utils import build_not_implemented_msg

class BaseValueFunction(object):

    def setup(self):
      pass

    def save(self, save_dir_path):
      pass

    def load(self, load_dir_path):
      pass

class BaseTabularActionValueFunction(BaseValueFunction):

    def predict_value(self, state, action):
        err_msg = build_not_implemented_msg(self, "predict_value")
        raise NotImplementedError(err_msg)

    def backup(self, state, action, backup_target, alpha):
        err_msg = build_not_implemented_msg(self, "backup")
        raise NotImplementedError(err_msg)

class BaseApproxActionValueFunction(BaseValueFunction):

    def predict_value(self, features):
        err_msg = build_not_implemented_msg(self, "predict_value")
        raise NotImplementedError(err_msg)

    def backup(self, features, backup_target, alpha):
        err_msg = build_not_implemented_msg(self, "backup")
        raise NotImplementedError(err_msg)


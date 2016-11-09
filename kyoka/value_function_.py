from kyoka.utils import build_not_implemented_msg

class BaseActionValueFunction(object):

    def predict_value(self, state, action):
        err_msg = build_not_implemented_msg(self, "predict_value")
        raise NotImplementedError(err_msg)

    def backup(self, state, action, backup_target, alpha):
        err_msg = build_not_implemented_msg(self, "backup")
        raise NotImplementedError(err_msg)

    def setup(self):
      pass

    def save(self, save_dir_path):
      pass

    def load(self, load_dir_path):
      pass


class BaseTabularActionValueFunction(BaseActionValueFunction):
    pass


class BaseApproxActionValueFunction(BaseActionValueFunction):

    def predict_value(self, state, action):
        return self.approx_predict_value(self.construct_features(state, action))

    def backup(self, state, action, backup_target, alpha):
        self.approx_backup(self.construct_features(state, action), backup_target, alpha)

    def construct_features(self, state, action):
        err_msg = build_not_implemented_msg(self, "construct_features")
        raise NotImplementedError(err_msg)

    def approx_predict_value(self, features):
        err_msg = build_not_implemented_msg(self, "approx_predict_value")
        raise NotImplementedError(err_msg)

    def approx_backup(self, features, backup_target, alpha):
        err_msg = build_not_implemented_msg(self, "approx_backup")
        raise NotImplementedError(err_msg)


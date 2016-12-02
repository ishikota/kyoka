import os

from kyoka.utils import build_not_implemented_msg, pickle_data, unpickle_data


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

    BASE_SAVE_FILE_NAME = "table_action_value_function_data.pickle"

    def generate_initial_table(self):
        err_msg = build_not_implemented_msg(self, "generate_initial_table")
        raise NotImplementedError(err_msg)

    def fetch_value_from_table(self, table, state, action):
        err_msg = build_not_implemented_msg(self, "fetch_value_from_table")
        raise NotImplementedError(err_msg)

    def insert_value_into_table(self, table, state, action, new_value):
        err_msg = build_not_implemented_msg(self, "insert_value_into_table")
        raise NotImplementedError(err_msg)

    def define_save_file_prefix(self):
        return ""

    def setup(self):
        self.table = self.generate_initial_table()

    def predict_value(self, state, action):
        return self.fetch_value_from_table(self.table, state, action)

    def save(self, save_dir_path):
        pickle_data(self._gen_table_data_file_path(save_dir_path), self.table)

    def load(self, load_dir_path):
        file_path = self._gen_table_data_file_path(load_dir_path)
        if not os.path.exists(file_path):
            raise IOError('The saved data of "TableActionValueFunction" is not found on [ %s ]' % load_dir_path)
        self.table = unpickle_data(file_path)

    def _gen_table_data_file_path(self, dir_path):
        return os.path.join(dir_path, self._gen_table_data_file_name())

    def _gen_table_data_file_name(self):
        prefix = self.define_save_file_prefix()
        if len(prefix) != 0: prefix += "_"
        return prefix + self.BASE_SAVE_FILE_NAME


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


import os

from kyoka.utils import build_not_implemented_msg, pickle_data, unpickle_data


class BaseActionValueFunction(object):
    """Base class of tabular and approximation action value function.

    The responsibility of action value function is to impelement two methods
    - predict_value
    - backup
    """

    def predict_value(self, state, action):
        """Predict the value of passed state-action pair.
        Returns:
            value: predicted value of passed state-action pair
        """
        err_msg = build_not_implemented_msg(self, "predict_value")
        raise NotImplementedError(err_msg)

    def backup(self, state, action, backup_target, alpha):
        """Update the value of passed state-action pair
        Args:
            state : state of state-action pair to update the value
            action : action of state-action pair to update the value
            backup_target : update the value by using this target which created from RL algorithm
            alpha : learning parameter passed from RL algorithm
        """
        err_msg = build_not_implemented_msg(self, "backup")
        raise NotImplementedError(err_msg)

    def setup(self):
      pass

    def save(self, save_dir_path):
      pass

    def load(self, load_dir_path):
      pass


class BaseTabularActionValueFunction(BaseActionValueFunction):
    """Base class of tabular value function used in RL algorithms

    property:
        table : the table to store the values. This property is initialized
                in setup method by using "generate_initial_table" method.
    """

    BASE_SAVE_FILE_NAME = "table_action_value_function_data.pickle"

    def generate_initial_table(self):
        """Initialize table to store the values of state-action pairs.
        Returns:
            table: this table is passed to "fetch_value_from_table" and
                   "insert_value_into_table" methods.
        """
        err_msg = build_not_implemented_msg(self, "generate_initial_table")
        raise NotImplementedError(err_msg)

    def fetch_value_from_table(self, table, state, action):
        """Define how to fetch the value of state-action pair from table.
        Args:
            table : current table object which initialzed by "generate_initial_table" method
            state : state of state-action pair to fetch the value from table
            action : action of state-action pair to fetch the value from table
        Returns:
            value : the value of state-action pair fetched from the table
        """
        err_msg = build_not_implemented_msg(self, "fetch_value_from_table")
        raise NotImplementedError(err_msg)

    def insert_value_into_table(self, table, state, action, new_value):
        """how to insert the new_item into table indexed by state-action pair

        This method directly update passed table by inserting new_value.
        (so thie method causes side-effect through table object)

        Args:
            table : table to insert the value (initialized by "generate_initial_table" method)
            state : state of state-action pair to index where to insert the new_value
            action: action of state-action pair to index where to insert the new_value
            new_value : new_value to insert into the table
        Returns:
            nothing : because directly update passed table object
        """
        err_msg = build_not_implemented_msg(self, "insert_value_into_table")
        raise NotImplementedError(err_msg)

    def define_save_file_prefix(self):
        """
        If you return "boo" then "self.save("some_dir")" will create
        "some_dir/boo_table_action_value_function_data.pickle"
        """
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
            raise IOError('The saved data of "TableActionValueFunction"' +
                          ' is not found on [ %s ]' % load_dir_path)
        self.table = unpickle_data(file_path)

    def _gen_table_data_file_path(self, dir_path):
        return os.path.join(dir_path, self._gen_table_data_file_name())

    def _gen_table_data_file_name(self):
        prefix = self.define_save_file_prefix()
        if len(prefix) != 0: prefix += "_"
        return prefix + self.BASE_SAVE_FILE_NAME


class BaseApproxActionValueFunction(BaseActionValueFunction):
    """Base class of approximation value function

    Child class needs to implement following 3 methods for
    "predict_value" and "backup" which is necessary for action value function.

    - construct_features : transform state to feature representation
    - approx_predict_value : predict value by using feature representation
    - approx_backup : backup target valie by using feature representation
    """

    def predict_value(self, state, action):
        return self.approx_predict_value(self.construct_features(state, action))

    def backup(self, state, action, backup_target, alpha):
        self.approx_backup(self.construct_features(state, action), backup_target, alpha)

    def construct_features(self, state, action):
        """Transform state to compact feature representation
        Args:
            state: state of state-action pair to transform
            action: action of state-action pair to transform
        Returns:
            features: features which represents passed state-action pair
        """
        err_msg = build_not_implemented_msg(self, "construct_features")
        raise NotImplementedError(err_msg)

    def approx_predict_value(self, features):
        """Predict value by using feature representation of state-action pair
        Args:
            features: transformed by "construct_features" method
        Returns:
            value : predict the value of state-action pair by using features
        """
        err_msg = build_not_implemented_msg(self, "approx_predict_value")
        raise NotImplementedError(err_msg)

    def approx_backup(self, features, backup_target, alpha):
        """Update value by using feature representation of state-action pair
        Args:
            features: transformed by "construct_features" method
            backup_target : update the value by using this target which created from RL algorithm
            alpha : learning parameter passed from RL algorithm
        """
        err_msg = build_not_implemented_msg(self, "approx_backup")
        raise NotImplementedError(err_msg)


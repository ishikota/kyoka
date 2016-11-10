import os

from kyoka.utils import pickle_data, unpickle_data
from kyoka.value_function_ import BaseTabularActionValueFunction
from kyoka.algorithm_.rl_algorithm import BaseRLAlgorithm, generate_episode

class MonteCarlo(BaseRLAlgorithm):

    def setup(self, task, policy, value_function):
        validate_value_function(value_function)
        super(MonteCarlo, self).setup(task, policy, value_function)

    def run_gpi_for_an_episode(self, task, policy, value_function):
        episode = generate_episode(task, policy, value_function)
        for idx, turn_info in enumerate(episode):
            state, action, _next_state, _reward = turn_info
            following_reward = self._calculate_following_state_reward(idx, episode)
            value_function.backup(state, action, following_reward, alpha="dummy")

    def _calculate_following_state_reward(self, current_turn, episode):
        following_turn_info = episode[current_turn:]
        following_reward = [reward for _, _, _, reward in following_turn_info]
        return sum(following_reward)

class MontCarloTabularActionValueFunction(BaseTabularActionValueFunction):

    SAVE_FILE_NAME = "montecarlo_update_counter.pickle"

    def setup(self):
        super(MontCarloTabularActionValueFunction, self).setup()
        self.update_counter = self.generate_initial_table()

    def define_save_file_prefix(self):
        return "montecarlo"

    def save(self, save_dir_path):
        super(MontCarloTabularActionValueFunction, self).save(save_dir_path)
        pickle_data(self._gen_update_counter_file_path(save_dir_path), self.update_counter)

    def load(self, load_dir_path):
        super(MontCarloTabularActionValueFunction, self).load(load_dir_path)
        if not os.path.exists(self._gen_update_counter_file_path(load_dir_path)):
            raise IOError('The saved data of "MonteCarlo" algorithm is not found in [ %s ]'% load_dir_path)
        self.update_counter = unpickle_data(self._gen_update_counter_file_path(load_dir_path))

    def backup(self, state, action, backup_target, alpha):
        update_count = self.fetch_value_from_table(self.update_counter, state, action)
        Q_value = self.fetch_value_from_table(self.table, state, action)
        new_value = self._calc_average_in_incremental_way(update_count, backup_target, Q_value)
        self.insert_value_into_table(self.table, state, action, new_value)
        self.insert_value_into_table(self.update_counter, state, action, update_count+1)

    def _calc_average_in_incremental_way(self, k, r, Q):
        return Q + 1.0 / (k + 1) * (r - Q)

    def _gen_update_counter_file_path(self, dir_path):
        return os.path.join(dir_path, self.SAVE_FILE_NAME)

def validate_value_function(value_function):
    if not isinstance(value_function, MontCarloTabularActionValueFunction):
        err_msg = 'MonteCarlo method requires you to use "table" type function.\
            (child class of [BaseTableStateValueFunction or BaseTableActionValueFunction])'
        raise TypeError(err_msg)

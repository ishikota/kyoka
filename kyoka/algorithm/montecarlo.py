import os

from kyoka.utils import pickle_data, unpickle_data, value_function_check
from kyoka.value_function import BaseTabularActionValueFunction, BaseApproxActionValueFunction
from kyoka.algorithm.rl_algorithm import BaseRLAlgorithm, generate_episode


class MonteCarlo(BaseRLAlgorithm):
    """Every-visit MonteCarlo method with supporting reward discounting,

    "Every-visit" means "using every state for update in an episode even if
    same state appeared in the episode".

    Algorithm is implemented based on the book "Reinforcement Learning: An Introduction"
    (reference : https://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf)

    - Algorithm -
    Initialize:
        T  <- your RL task
        PI <- Policy used to generate episode
        Q  <- action value function
    Repeat until computational budge runs out:
        generate an episode of T by following policy PI
        for each state-action pair (S, A)  appeared in the episode:
            G <- sum of rewards gained after state S (discounted if gamma < 1)
            Q(S, A) <- average G of S sampled ever
    """

    def __init__(self, gamma=1):
        """
        If you want to discount future reward then set gamma < 1.

        For example, we have an apisode like this
        episode :
            (state0, action0) -> reward0 ->
            (state1, action1) -> reward1 ->
            (state2, action2) -> reward2 -> finish

        then reward discounting is done like this
            reward_sum_from_state0 = reward0 + gamma * reward1 + gamma**2 reward2

        Args:
            gamma : discount factor of reward. default=1. 0 < gamma <= 1.
         """
        self.gamma = gamma

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
        return sum([self.__discount(step, reward) for step, reward in enumerate(following_reward)])

    def __discount(self, step, reward):
        return self.gamma ** step * reward

class MonteCarloTabularActionValueFunction(BaseTabularActionValueFunction):
    """Tabular action value function for MonteCarlo method.

    Backup target passed from MonteCarlo is G(reward sum of state S).
    So backup is done just averaging G of S sampled ever.

    Calculation of average is implemented by memory efficient way.
    ("_calc_average_in_incremental_way" is the method calculates average")
    """

    SAVE_FILE_NAME = "montecarlo_update_counter.pickle"

    def setup(self):
        super(MonteCarloTabularActionValueFunction, self).setup()
        self.update_counter = self.generate_initial_table()

    def define_save_file_prefix(self):
        return "montecarlo"

    def save(self, save_dir_path):
        super(MonteCarloTabularActionValueFunction, self).save(save_dir_path)
        pickle_data(self._gen_update_counter_file_path(save_dir_path), self.update_counter)

    def load(self, load_dir_path):
        super(MonteCarloTabularActionValueFunction, self).load(load_dir_path)
        if not os.path.exists(self._gen_update_counter_file_path(load_dir_path)):
            raise IOError(
                    'The saved data of "MonteCarlo" algorithm is not found in [ %s ]' %
                    load_dir_path)
        self.update_counter = unpickle_data(self._gen_update_counter_file_path(load_dir_path))

    def backup(self, state, action, backup_target, alpha):
        update_count = self.fetch_value_from_table(self.update_counter, state, action)
        Q_value = self.fetch_value_from_table(self.table, state, action)
        new_value = self._calc_average_in_incremental_way(update_count, backup_target, Q_value)
        self.insert_value_into_table(self.table, state, action, new_value)
        self.insert_value_into_table(self.update_counter, state, action, update_count+1)

    def _calc_average_in_incremental_way(self, k, r, Q):
        """Memory efficient implementation to calculate average"""
        return Q + 1.0 / (k + 1) * (r - Q)

    def _gen_update_counter_file_path(self, dir_path):
        return os.path.join(dir_path, self.SAVE_FILE_NAME)

class MonteCarloApproxActionValueFunction(BaseApproxActionValueFunction):
    """Approximation action value function for MonteCarlo method.
    There is no additional method from base class to use MonteCarlo method.

    Backup target passed from MonteCarlo is G(reward sum of state S).
    So backup should be done to approximate average of G of S sampled ever.
    """
    pass

def validate_value_function(value_function):
    value_function_check("MonteCarlo",
            [MonteCarloTabularActionValueFunction, MonteCarloApproxActionValueFunction],
            value_function)


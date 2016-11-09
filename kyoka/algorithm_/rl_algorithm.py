from kyoka.utils import build_not_implemented_msg
from kyoka.policy_ import EpsilonGreedyPolicy
from kyoka.callback_ import EpsilonAnnealer
from kyoka.callback_ import WatchIterationCount

def generate_episode(task, value_function, policy):
    state = task.generate_initial_state()
    episode = []
    while not task.is_terminal_state(state):
        action = policy.choose_action(task, value_function, state)
        next_state = task.transit_state(state, action)
        reward = task.calculate_reward(next_state)
        episode.append((state, action, next_state, reward))
        state = next_state
    return episode

class BaseRLAlgorithm(object):

    def setup(self, task, policy, value_function):
        self.task = task
        self.value_function = value_function
        self.value_function.setup()
        self.policy = policy

    def save(self, save_dir_path):
        self.value_function.save(save_dir_path)
        self.save_algorithm_state(save_dir_path)

    def load(self, load_dir_path):
        self.value_function.load(load_dir_path)
        self.load_algorithm_state(load_dir_path)

    def save_algorithm_state(self, save_dir_path):
        pass

    def load_algorithm_state(self, load_dir_path):
        pass

    def run_gpi_for_an_episode(self, task, policy, value_function):
        err_msg = build_not_implemented_msg(self, "run_gpi_for_an_episode")
        raise NotImplementedError(err_msg)

    def run_gpi(self, nb_iteration, callbacks=None, verbose=1):
        self.__check_setup_call()
        default_finish_rule = WatchIterationCount(nb_iteration, verbose)
        callbacks = self.__setup_callbacks(default_finish_rule, callbacks)
        [callback.before_gpi_start(self.task, self.value_function) for callback in callbacks]

        iteration_counter = 1
        while True:
            [callback.before_update(iteration_counter, self.task, self.value_function) for callback in callbacks]
            self.run_gpi_for_an_episode(self.task, self.policy, self.value_function)
            [callback.after_update(iteration_counter, self.task, self.value_function) for callback in callbacks]
            for finish_rule in callbacks:
                if finish_rule.interrupt_gpi(iteration_counter, self.task, self.value_function):
                    [callback.after_gpi_finish(self.task, self.value_function) for callback in callbacks]
                    if finish_rule != default_finish_rule:
                        default_finish_rule.log(default_finish_rule.generate_finish_message(iteration_counter))
                    return
            iteration_counter += 1


    def __check_setup_call(self):
        if not all([hasattr(self, attr) for attr in ["task", "value_function", "policy"]]):
            raise Exception('You need to call "setup" method before calling "run_gpi" method.')

    def __setup_callbacks(self, default_finish_rule, user_callbacks):
        user_callbacks = self.__wrap_item_if_single(user_callbacks)
        default_callbacks = [default_finish_rule]
        if isinstance(self.policy, EpsilonGreedyPolicy) and self.policy.do_annealing:
            default_callbacks.append(EpsilonAnnealer(self.policy))
        return default_callbacks + user_callbacks

    def __wrap_item_if_single(self, item):
        if item is None: item = []
        if not isinstance(item, list): item = [item]
        return item


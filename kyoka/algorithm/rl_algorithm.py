from kyoka.utils import build_not_implemented_msg
from kyoka.policy import EpsilonGreedyPolicy
from kyoka.callback import EpsilonAnnealer, WatchIterationCount


def generate_episode(task, policy, value_function):
    """Utility method to generate an episode by following passed
    policy and value_function

    Args:
        task: Task object which represents some RL problem
        policy : generate episode by following this policy
        value_function : choose action by following this value function
    Returns:
        episode : Array of experience (tuple of state, action, next_state, reward).
                  Last item(experience) of next_state(3rd element of experience)
                  must be terminal state.
    """
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
    """Base class for all of offline RL algorithms.

    "run_gpi" is the method for training (learning value function).
    But its core logic of how to update value function is delegated to
    abstract method "run_gpi_for_an_episode".
    So the role of child class is implement this abstract method.

    Do not forget to call setup method before calling "run_gpi" method.
    (If you forget, Exception is raised on "run_gpi" call.)
    """

    def setup(self, task, policy, value_function):
        """Set items which necesarry for training"""
        self.task = task
        self.value_function = value_function
        self.value_function.setup()
        self.policy = policy

    def save(self, save_dir_path):
        """Save value function and state of algorithm if exist.
        Args:
            save_dir_path: items are saved under this directory
        """
        self.value_function.save(save_dir_path)
        self.save_algorithm_state(save_dir_path)

    def load(self, load_dir_path):
        """load saved items from directory where save method executed.
        Args:
            load_dir_path: this should be the path you passed save method
                           as save_dir_path
        """
        self.value_function.load(load_dir_path)
        self.load_algorithm_state(load_dir_path)

    def save_algorithm_state(self, save_dir_path):
        """If algorithm uses some state variables in the training save it.
        Args:
            save_dir_path: items are saved under this directory
        """
        pass

    def load_algorithm_state(self, load_dir_path):
        """load saved state of algorithm from directory where
        "save_algorithm_state" method executed.
        Args:
            load_dir_path: this should be the path you passed
                           "save_algorithm_state" as save_dir_path
        """
        pass

    def run_gpi_for_an_episode(self, task, policy, value_function):
        """Define how to update value function for an episode."""
        err_msg = build_not_implemented_msg(self, "run_gpi_for_an_episode")
        raise NotImplementedError(err_msg)

    def run_gpi(self, nb_iteration, callbacks=None, verbose=1):
        """Run GPI(Generalized Policy Iteration) to improve value function
        Args:
            nb_iteration: run GPI for nb_iteration times
            callbacks: These callbacks are invoked during GPI.
                       callback objects (objects which inherit
                       "kyoka.callbacks.BaseCallback").
                       Array of callback objects or a callback object is acceptable.
            verbose: verbose > 0 logs progress of training
        """
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
        """Raise exception with message if run_gpi is called without setup"""
        if not all([hasattr(self, attr) for attr in ["task", "value_function", "policy"]]):
            raise Exception('You need to call "setup" method before calling "run_gpi" method.')

    def __setup_callbacks(self, default_finish_rule, user_callbacks):
        """Add EpsilonAnnealer if needed and put default_callback
        at top of callback list.
        """
        user_callbacks = self.__wrap_item_if_single(user_callbacks)
        default_callbacks = [default_finish_rule]
        if isinstance(self.policy, EpsilonGreedyPolicy) and self.policy.do_annealing:
            default_callbacks.append(EpsilonAnnealer(self.policy))
        return default_callbacks + user_callbacks

    def __wrap_item_if_single(self, item):
        """Make single callback object as single callback list"""
        if item is None: item = []
        if not isinstance(item, list): item = [item]
        return item


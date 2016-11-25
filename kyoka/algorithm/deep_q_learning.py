import os
import random

from kyoka.utils import pickle_data, unpickle_data, value_function_check, build_not_implemented_msg
from kyoka.policy import GreedyPolicy, EpsilonGreedyPolicy
from kyoka.value_function import BaseApproxActionValueFunction
from kyoka.algorithm.rl_algorithm import BaseRLAlgorithm, generate_episode

class DeepQLearning(BaseRLAlgorithm):

    SAVE_FILE_NAME = "dq_replay_memory.pickle"

    def __init__(self, gamma=0.99, N=1000000, C=10000, minibatch_size=32, replay_start_size=50000):
        self.gamma = gamma
        self.replay_memory = ExperienceReplay(max_size=N)
        self.C = C
        self.minibatch_size = minibatch_size
        self.replay_start_size = replay_start_size
        self.reset_step_counter = 0

    def setup(self, task, policy, value_function):
        validate_value_function(value_function)
        super(DeepQLearning, self).setup(task, policy, value_function)
        initialize_replay_memory(task, value_function, self.replay_memory, self.replay_start_size)
        self.greedy_policy = GreedyPolicy()

    def run_gpi_for_an_episode(self, task, policy, value_function):
        value_function.use_target_network(False)
        state = task.generate_initial_state()

        while not task.is_terminal_state(state):
            action = policy.choose_action(task, value_function, state)
            next_state = task.transit_state(state, action)
            reward = task.calculate_reward(next_state)
            self.replay_memory.store_transition(state, action, reward, next_state)
            state = next_state

            experience_minibatch = self.replay_memory.sample_minibatch(self.minibatch_size)
            backup_minibatch = self._gen_backup_minibatch(task, self.greedy_policy, value_function, experience_minibatch)
            value_function.backup_on_minibatch(value_function.q_network, backup_minibatch)

            if self.reset_step_counter >= self.C:
                value_function.reset_target_network()
                self.reset_step_counter = 0
            else:
                self.reset_step_counter += 1

    def save_algorithm_state(self, save_dir_path):
        state = (
                self.gamma, self.replay_memory.dump(), self.C, self.minibatch_size,
                self.replay_start_size, self.reset_step_counter
                )
        pickle_data(self._gen_replay_memory_save_path(save_dir_path), state)

    def load_algorithm_state(self, load_dir_path):
        state = unpickle_data(self._gen_replay_memory_save_path(load_dir_path))
        self.gamma, replay_memory_serial, self.C, self.minibatch_size,\
                self.replay_start_size, self.reset_step_counter = state
        self.greedy_policy = GreedyPolicy()
        new_replay_memory = ExperienceReplay(max_size=1)
        new_replay_memory.load(replay_memory_serial)
        self.replay_memory = new_replay_memory

    def _gen_backup_minibatch(self, task, greedy_policy, value_function, experience_minibatch):
        value_function.use_target_network(True)
        backup_minibatch = [self._gen_backup_data(task, greedy_policy, value_function, experience)\
                for experience in experience_minibatch]
        value_function.use_target_network(False)
        return backup_minibatch

    def _gen_backup_data(self, task, greedy_policy, value_function, experience):
        state, action, reward, next_state = experience
        greedy_action = choose_action(task, greedy_policy, value_function, next_state)
        greedy_Q_value = predict_value(value_function, next_state, greedy_action)
        backup_target = reward + self.gamma * greedy_Q_value
        return (state, action, backup_target)

    def _gen_replay_memory_save_path(self, dir_path):
        return os.path.join(dir_path, self.SAVE_FILE_NAME)

class DeepQLearningApproxActionValueFunction(BaseApproxActionValueFunction):

    def initialize_network(self):
        err_msg = build_not_implemented_msg(self, "initialize_network")
        raise NotImplementedError(err_msg)

    def deepcopy_network(self, q_network):
        err_msg = build_not_implemented_msg(self, "deepcopy_network")
        raise NotImplementedError(err_msg)

    def predict_value_by_network(self, network, state, action):
        err_msg = build_not_implemented_msg(self, "predict_value_by_network")
        raise NotImplementedError(err_msg)

    def backup_on_minibatch(self, q_network, backup_minibatch):
        err_msg = build_not_implemented_msg(self, "backup_on_minibatch")
        raise NotImplementedError(err_msg)

    def save_networks(self, q_network, q_hat_network, save_dir_path):
        err_msg = build_not_implemented_msg(self, "save_networks")
        raise NotImplementedError(err_msg)

    def load_networks(self, load_dir_path):
        err_msg = build_not_implemented_msg(self, "load_networks")
        raise NotImplementedError(err_msg)


    def setup(self):
        self.q_network = self.initialize_network()
        self.reset_target_network()
        self.use_target_network_flg = False

    def use_target_network(self, use_target_network):
        self.use_target_network_flg = use_target_network

    def predict_value(self, state, action):
        network = self.q_hat_network if self.use_target_network_flg else self.q_network
        return self.predict_value_by_network(network, state, action)

    def reset_target_network(self):
        self.q_hat_network = self.deepcopy_network(self.q_network)

    def save(self, save_dir_path):
        self.save_networks(self.q_network, self.q_hat_network, save_dir_path)

    def load(self, load_dir_path):
        self.q_network, self.q_hat_network = self.load_networks(load_dir_path)


class ExperienceReplay(object):

    def __init__(self, max_size):
        self.max_size = max_size
        self.queue = []

    def store_transition(self, state, action, reward, next_state):
        if len(self.queue) >= self.max_size:
            self.queue.pop(0)
        self.queue.append((state, action, reward, next_state))

    def sample_minibatch(self, minibatch_size):
        return random.sample(self.queue, minibatch_size)

    def dump(self):
        return (self.max_size, self.queue)

    def load(self, serial):
        self.max_size, self.queue = serial

def initialize_replay_memory(task, value_function, replay_memory, start_size):
    random_policy = EpsilonGreedyPolicy(eps=1.0)
    while len(replay_memory.queue) < start_size:
        for state, action, next_state, reward in generate_episode(task, random_policy, value_function):
            replay_memory.store_transition(state, action, reward, next_state)
            if len(replay_memory.queue) >= start_size: return

ACTION_ON_TERMINAL_FLG = "action_on_terminal"

def choose_action(task, policy, value_function, state):
    if task.is_terminal_state(state):
        return ACTION_ON_TERMINAL_FLG
    else:
        return policy.choose_action(task, value_function, state)

def predict_value(value_function, next_state, next_action):
    if ACTION_ON_TERMINAL_FLG == next_action:
        return 0
    else:
        return value_function.predict_value(next_state, next_action)

def validate_value_function(value_function):
    value_function_check("DeepQLearning",\
            [DeepQLearningApproxActionValueFunction],\
            value_function)


import os

from kyoka.utils import value_function_check
from kyoka.policy_ import GreedyPolicy
from kyoka.value_function_ import BaseTabularActionValueFunction, BaseApproxActionValueFunction
from kyoka.algorithm_.rl_algorithm import BaseRLAlgorithm

class QLearning(BaseRLAlgorithm):

    def __init__(self, alpha=0.1, gamma=0.9):
        self.alpha = alpha
        self.gamma = gamma

    def setup(self, task, policy, value_function):
        validate_value_function(value_function)
        super(QLearning, self).setup(task, policy, value_function)
        self.greedy_policy = GreedyPolicy()

    def run_gpi_for_an_episode(self, task, policy, value_function):
        state = task.generate_initial_state()
        action = policy.choose_action(task, value_function, state)
        while not task.is_terminal_state(state):
            next_state = task.transit_state(state, action)
            next_action = choose_action(task, policy, value_function, next_state)
            reward = task.calculate_reward(next_state)
            greedy_action = choose_action(task, self.greedy_policy, value_function, next_state)
            greedy_Q_value = predict_value(value_function, next_state, greedy_action)
            backup_target = reward + self.gamma * greedy_Q_value
            value_function.backup(state, action, backup_target, self.alpha)
            state, action = next_state, next_action

class QLearningTabularActionValueFunction(BaseTabularActionValueFunction):

    def define_save_file_prefix(self):
        return "q_learning"

    def backup(self, state, action, backup_target, alpha):
        Q_value = self.predict_value(state, action)
        new_Q_value = Q_value + alpha * (backup_target - Q_value)
        self.insert_value_into_table(self.table, state, action, new_Q_value)

class BaseQLearningApproxActionValueFunction(BaseApproxActionValueFunction):
    pass

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
    value_function_check("QLearning",\
            [QLearningTabularActionValueFunction, BaseQLearningApproxActionValueFunction],\
            value_function)


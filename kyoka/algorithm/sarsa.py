import os

from kyoka.utils import value_function_check
from kyoka.value_function import BaseTabularActionValueFunction, BaseApproxActionValueFunction
from kyoka.algorithm.rl_algorithm import BaseRLAlgorithm, generate_episode


class Sarsa(BaseRLAlgorithm):
    """Basic "on-policy" Temporal-Difference Learning method.

    "on-policy" indicates that this method uses same policy when
    "select action during the episode" and "create backup target".
    (backup target is the target value used when training value function)

    Algorithm is implemented based on the book "Reinforcement Learning: An Introduction"
    (reference : https://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf)

    - Algorithm -
    Initialize
        T  <- your RL task
        PI <- policy used in the algorithm
        Q  <- action value function
        a  <- learning rate (alpha)
        g  <- discounting factor (gamma)
    Repeat until computational budge runs out:
        S <- generate initial state of task T
        A <- choose action at S by following policy PI
        Repeat until S is terminal state:
            S' <- next state of S after taking action A
            R <- reward gained by taking action A at state S
            A' <- next action at S' by following policy PI
            Q(S, A) <- Q(S, A) + a * [ R + g * Q(S', A') - Q(S, A)]
            S, A <- S', A'
    """

    def __init__(self, alpha=0.1, gamma=0.9):
        """
        Args:
            alpha: learning rate. default=0.1. 0 < alpha <= 1
            gamma: discounting factor. default=0.9. 0 < gamma <= 1
        """
        self.alpha = alpha
        self.gamma = gamma

    def setup(self, task, policy, value_function):
        validate_value_function(value_function)
        super(Sarsa, self).setup(task, policy, value_function)

    def run_gpi_for_an_episode(self, task, policy, value_function):
        state = task.generate_initial_state()
        action = policy.choose_action(task, value_function, state)
        while not task.is_terminal_state(state):
            next_state = task.transit_state(state, action)
            next_action = choose_action(task, policy, value_function, next_state)
            reward = task.calculate_reward(next_state)
            next_Q_value = predict_value(value_function, next_state, next_action)
            backup_target = reward + self.gamma * next_Q_value
            value_function.backup(state, action, backup_target, self.alpha)
            state, action = next_state, next_action

class SarsaTabularActionValueFunction(BaseTabularActionValueFunction):
    """Tabular action value function for Sarsa.

    Backup target(TD error) passed from Sarsa is "R + g * Q(S', A')".
    So backup is done by following next equation.
        Q(S, A) <- Q(S, A) + a * [ R + g * Q(S', A') - Q(S, A)]
    """

    def define_save_file_prefix(self):
        return "sarsa"

    def backup(self, state, action, backup_target, alpha):
        Q_value = self.predict_value(state, action)
        new_Q_value = Q_value + alpha * (backup_target - Q_value)
        self.insert_value_into_table(self.table, state, action, new_Q_value)

class SarsaApproxActionValueFunction(BaseApproxActionValueFunction):
    """Approximation action value function for Sarsa.
    There is no additional method from base class to use QLearning.

    Backup target(TD error) passed from Sarsa is "R + g * Q(S', A')".
    So backup should be done to approximate next update.
        Q(S, A) <- Q(S, A) + a * [ R + g * Q(S', A') - Q(S, A)]
    """
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
    value_function_check("Sarsa",
            [SarsaTabularActionValueFunction, SarsaApproxActionValueFunction],
            value_function)


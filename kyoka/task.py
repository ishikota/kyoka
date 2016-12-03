from utils import build_not_implemented_msg


class BaseTask(object):
    """Base class to represent your problem as RL task

    RL algorithm requires following 5 methods to learn ActionValueFunction from some task.
    - generate_initial_state
    - is_terminal_state
    - transit_state
    - generate_possible_actions
    - calculate_reward
    """

    def generate_inital_state(self):
        """Define start state of your problem.
        Returns:
            initial_state: representation of initial state of your problem.
        """
        err_msg = build_not_implemented_msg(self, "generate_inital_state")
        raise NotImplementedError(err_msg)

    def is_terminal_state(self, state):
        """Define when is the finish of episode (your problem)
        Returns:
            is_terminal(boolean): True if your problem is finished in the state.
        """
        err_msg = build_not_implemented_msg(self, "is_terminal_state")
        raise NotImplementedError(err_msg)

    def transit_state(self, state, action):
        """Define the rule of state transition in your problem
        Args:
            state: current state which agent exists.
            action: action choosed in current state
        Returns:
            next_state: next state after applied passed action to current state
        """
        err_msg = build_not_implemented_msg(self, "transit_state")
        raise NotImplementedError(err_msg)

    def generate_possible_actions(self, state):
        """Define what action is possible in each state
        Args:
            state: we want to know legal actions on this state
        Returns:
            actions: actions which is legal in passed state
        """
        err_msg = build_not_implemented_msg(self, "generate_possible_actions")
        raise NotImplementedError(err_msg)

    def calculate_reward(self, state):
        """Define reward of each state
        Args:
            state: we want to know the reward of this state
        Returns:
            reward(scalar): the reward of passed state
        """
        err_msg = build_not_implemented_msg(self, "calculate_reward")
        raise NotImplementedError(err_msg)


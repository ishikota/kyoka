import os
import random

from kyoka.utils import pickle_data, unpickle_data, value_function_check, build_not_implemented_msg
from kyoka.policy import GreedyPolicy, EpsilonGreedyPolicy
from kyoka.value_function import BaseApproxActionValueFunction
from kyoka.algorithm.rl_algorithm import BaseRLAlgorithm, generate_episode


class DeepQLearning(BaseRLAlgorithm):
    """deep Q-learning method using experience replay proposed in
    the paper "Human-level control through deep reinforcement learning".

    deep Q-learning arranges Q-Learning in 2 points to improve and
    get stability of learing process.
    - Use mini-batch of experience to update value function.
    - Separate value function used to "create backup target".

    Algorithm is implemented based on the original paper.
    (reference: http://www.nature.com/nature/journal/v518/n7540/abs/nature14236.html)

    - Algorithm -
    Parameter
        g <- discounting factor (gamma)
        N <- capacity of replay memory
        C <- interval to sync Q' with Q
        minibatch_size <- size of minibatch used to train Q
        replay_start_size <- initial size of replay memory. Fill D
            with this number of experiences which created by random policy.
            This procedure is done in setup phase.

    Initialize
        T  <- your RL task
        PI <- policy used to "select action during the episode"
        Q  <- approximate action value function (ex. use neural network)
        Q' <- separated value function created by deepcopy Q
        D <- replay memory of capacity N filled with experience of initial size
    Repeat until computational budget runs out:
        S <- generate initial state of task T
        A <- choose action at S by following PI
        Repeat until S is terminal state:
            S' <- next state of S after taking action A
            R <- reward gained by taking action A at state S
            A' <- next action at S' by following policy PI
            append experience (S,A,R,S') to D
            MB <- sample "minibatch_size" experiences from D
            BT <- transform minibatch of experiences into backup targets
            (BT = [r + g * Q(s', GA) for s,a,r,s' in MB], GA=greedy action at s')
            Update Q by using BT (minibatch of backup targets)
            Every C step: Q' <- Q (ex. deepcopy weights of Q to Q')
            S, A <- S', A'
    """

    SAVE_FILE_NAME = "dq_replay_memory.pickle"

    def __init__(self, gamma=0.99, N=1000000, C=10000,
            minibatch_size=32, replay_start_size=50000):
        """
        Args:
            g <- discounting factor (gamma)
            N <- capacity of replay memory
            C <- interval to sync Q' with Q
            minibatch_size <- size of minibatch used to train Q
            replay_start_size <- initial size of replay memory.
        """

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
            backup_minibatch = self._gen_backup_minibatch(
                    task, self.greedy_policy, value_function, experience_minibatch)
            value_function.backup_on_minibatch(value_function.q_network, backup_minibatch)

            if self.reset_step_counter >= self.C:
                value_function.reset_target_network()
                self.reset_step_counter = 0
            else:
                self.reset_step_counter += 1

    def save_algorithm_state(self, save_dir_path):
        """Save initial params, replay memory and counter for sync Q' with Q"""
        state = (
                self.gamma, self.replay_memory.dump(), self.C, self.minibatch_size,
                self.replay_start_size, self.reset_step_counter
                )
        pickle_data(self._gen_replay_memory_save_path(save_dir_path), state)

    def load_algorithm_state(self, load_dir_path):
        """Load initial params, replay memory and counter for sync Q' with Q"""
        state = unpickle_data(self._gen_replay_memory_save_path(load_dir_path))
        (self.gamma, replay_memory_serial, self.C, self.minibatch_size,
                self.replay_start_size, self.reset_step_counter) = state
        self.greedy_policy = GreedyPolicy()
        new_replay_memory = ExperienceReplay(max_size=1)
        new_replay_memory.load(replay_memory_serial)
        self.replay_memory = new_replay_memory

    def _gen_backup_minibatch(self, task, greedy_policy, value_function, experience_minibatch):
        """Create minibatch of backup targets from minibatch of experiences
        Returns
            backup_minibatch : minibatch of training data for value function.
                               It's array of learning data which is tuple of
                               (state, action, backup_target).
                               Most of the case value function is trained by
                               using MSE between Q(state, action) and backup_target.
        """
        value_function.use_target_network(True)
        backup_minibatch = [
                self._gen_backup_data(task, greedy_policy, value_function, experience)
                for experience in experience_minibatch]
        value_function.use_target_network(False)
        return backup_minibatch

    def _gen_backup_data(self, task, greedy_policy, value_function, experience):
        """Transform experience into backup targets in training data format
        Returns
            learning data: tuple of (state, action, backup_target).
                           value function receives minibatch of this tuples and
                           train value function maybe like below.
                           MSE <- ( Q(state, action) - backup_target )^2
        """
        state, action, reward, next_state = experience
        greedy_action = choose_action(task, greedy_policy, value_function, next_state)
        greedy_Q_value = predict_value(value_function, next_state, greedy_action)
        backup_target = reward + self.gamma * greedy_Q_value
        return (state, action, backup_target)

    def _gen_replay_memory_save_path(self, dir_path):
        return os.path.join(dir_path, self.SAVE_FILE_NAME)

class DeepQLearningApproxActionValueFunction(BaseApproxActionValueFunction):
    """Base class of Approximation value function for deep Q-learing.

    Child class required to implement following 6 methods.
    - initialize_network
    - deepcopy_network
    - predict_value_by_network
    - backup_on_minibatch
    - save_networks (if you want save/load feature)
    - load_networks (if you want save/load feature)
    """

    def initialize_network(self):
        """Initialize action value function Q
        Returns:
            q_network: Item returned here is used as Q value function.
                       Most of the case, you initialize neuralnetwork here
                       and return it.
                       You can access this item by "value_function.q_network".
        """
        err_msg = build_not_implemented_msg(self, "initialize_network")
        raise NotImplementedError(err_msg)

    def deepcopy_network(self, q_network):
        """Deepcopy Q to initialize and sync separated network Q'
        Args:
            q_network: Q value function initialized by "initialize_network"
        Returns:
            q_hat_network: deepcopy of q_network which passed as argument.
        """
        err_msg = build_not_implemented_msg(self, "deepcopy_network")
        raise NotImplementedError(err_msg)

    def predict_value_by_network(self, network, state, action):
        """Define how to predict action value by Q and Q' which you defined
        Args:
            network: q_network or q_hat_network which initialized by
                     "initialize_network" or "deepcopy_network"
            state: state of state-action pair to predict value
            action: action of state-action pair to predict value
        Returns:
            predicted_value: prediction value of action by passed network
        """
        err_msg = build_not_implemented_msg(self, "predict_value_by_network")
        raise NotImplementedError(err_msg)

    def backup_on_minibatch(self, q_network, backup_minibatch):
        """Define how to train Q network which you defined
        Args:
            q_network: Q value function initialized by "initialize_network"
            backup_minibatch : minibatch of training data for Q. It's array of
                               learning data which is tuple of
                               (state, action, backup_target).
                               Most of the case value function is trained by
                               using MSE between Q(state, action) and backup_target.
        """
        err_msg = build_not_implemented_msg(self, "backup_on_minibatch")
        raise NotImplementedError(err_msg)

    def save_networks(self, q_network, q_hat_network, save_dir_path):
        """Save Q and Q' under passed directory
        Args:
            q_network: Q value function initialized by "initialize_network"
            q_hat_network: Q' deepcopied by "deepcopy_network"
            save_dir_path: path to directory to save. Load method would be
                           called on this directory.
        """
        err_msg = build_not_implemented_msg(self, "save_networks")
        raise NotImplementedError(err_msg)

    def load_networks(self, load_dir_path):
        """Load Q and Q' from passed directory
        Args:
            load_dir_path: path to directory which you executed "save_networks"
        Returns:
            networks: tuple of Q and Q' which loaded from passed directory.
                      Return loaded networks like below.

                      q_network = # load some way
                      q_hat_network = # load some way
                      return q_network, q_hat_network
        """
        err_msg = build_not_implemented_msg(self, "load_networks")
        raise NotImplementedError(err_msg)


    def setup(self):
        """Initialize Q and Q'"""
        self.q_network = self.initialize_network()
        self.reset_target_network()
        self.use_target_network_flg = False

    def use_target_network(self, use_target_network):
        """Switch value function for prediction between Q and Q'
        Use Q on prediction if use_target_network=True else Q'
        """
        self.use_target_network_flg = use_target_network

    def predict_value(self, state, action):
        """Predict value of state-action pair with Q or Q'
        Main logic of prediction is delegated to abstract method
        "predict_value_by_network".

        Switch value function by watching self.use_target_network_flg.
        You can switch this flg by using "use_target_network_flg(True/False)".
        """
        network = self.q_hat_network if self.use_target_network_flg else self.q_network
        return self.predict_value_by_network(network, state, action)

    def reset_target_network(self):
        """Sync Q' with Q by calling user-defined method "deepcopy_network" """
        self.q_hat_network = self.deepcopy_network(self.q_network)

    def save(self, save_dir_path):
        """Save networks under passed directory. Main logic is delegated to
        abstract method "save_networks".
        """
        self.save_networks(self.q_network, self.q_hat_network, save_dir_path)

    def load(self, load_dir_path):
        """Load networks from passed directory and set them as property. Main
        logic is delegated to abstract method "load_networks".
        """
        self.q_network, self.q_hat_network = self.load_networks(load_dir_path)


class ExperienceReplay(object):
    """Implementation of ExperienceReplayMemory"""

    def __init__(self, max_size):
        """
        Args:
            max_size: capacity of replay memory. If size of memory exceeds
                      after store_transition, old item will be poped from
                      memory. (FIFO)
        """
        self.max_size = max_size
        self.queue = []

    def store_transition(self, state, action, reward, next_state):
        """Store new experience and pop old item if it reaches max size"""
        if len(self.queue) >= self.max_size:
            self.queue.pop(0)
        self.queue.append((state, action, reward, next_state))

    def sample_minibatch(self, minibatch_size):
        """Return array of experiences with specified size by sampling
        from replay memory at random.
        """
        return random.sample(self.queue, minibatch_size)

    def dump(self):
        return (self.max_size, self.queue)

    def load(self, serial):
        self.max_size, self.queue = serial

def initialize_replay_memory(task, value_function, replay_memory, start_size):
    """Fill passed replay memory with specified size of experience. Experience
    is created by generating episode with random policy until expected size
    of experience is corrected.
    """
    random_policy = EpsilonGreedyPolicy(eps=1.0)
    while len(replay_memory.queue) < start_size:
        episode = generate_episode(task, random_policy, value_function)
        for state, action, next_state, reward in episode:
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
    value_function_check("DeepQLearning",
            [DeepQLearningApproxActionValueFunction],
            value_function)


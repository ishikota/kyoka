import random
import math

from kyoka.utils import build_not_implemented_msg
from kyoka.task import BaseTask


class BaseMCTS(object):
    """Base class to implement MonteCarloTreeSearch method.

    This class implements base logic of MonteCarloTreeSearch algorithm like
    building and traversing game tree, backpropagation of simulation result.
    (These base procedure is implemented in "planning" method)

    But "how to build node of game tree from state" and "how to calculate
    value of edge(action)" is abstracted. (Because these points are strongly
    depend on the task to solve.)
    So you need to implement these points by creating your own
    Node(child class of BaseNode) and Edge(child class of BaseEdge) classes.
    And return that Node in "generate_node_from_state" method of this class.

    We prepared implmentation of UCT search as UCTNode class. If you want to
    use UCT search, return UCTNode in "generate_node_from_state" by
    initializing node with passed state like below.

        def generate_node_from_state(self, state):
            return UCTNode(self.task, state)

    If you want to use algorithm not UCT search, please implement your Node and
    Edge class referring the imeplemntation of UCTNode and UCTEdge classes.

    Algorithm is implemented based on the papaer
    "A Survey of Monte Carlo Tree Search Methods"
    (reference: http://ieeexplore.ieee.org/document/6145622/)
    """

    def __init__(self, task):
        """
        Args:
            task: task object to apply MCTS method
        """
        self.task = task
        self.playout_policy = random_playout
        self.last_calculated_tree = None
        self.finish_rule = None

    def generate_node_from_state(self, state):
        """Transform state of the task into Node(child class of BaseNode).

        Args:
            state: you return node of tree which represents this state
        Returns:
            node: node which represents passed state. Node must be child of
                  BaseNode class in this module.
        """
        err_msg = build_not_implemented_msg(self, "generate_node_from_state")
        raise NotImplementedError(err_msg)

    def set_playout_policy(self, policy):
        """Define how to play simulation(playout). Default is random play.

        In default, "random_playout" method is set as playout policy.
        If you want to update this behavior create your own playout policy
        and set it by this method.
        The format of playout policy method to pass this method would be
        like this.

            def my_playout_policy(self, task, leaf_node):
                current_state = leaf_node.state
                while not task.is_terminal_state(state):
                    state = current_node.state
                    action = choose_action_by_my_logic(state)
                    state = task.transit_state(state, action)
                return task.calculate_reward(state)

        And set above method like this.
            my_mcts.set_playout_policy(my_playout_policy)

        Args:
            policy: the method to run a simulation. this is used in _playout
                    method of BaseMCTS class.
        """
        self.playout_policy = policy

    def set_finish_rule(self, finish_rule):
        """Set finish rule of planning if you want to call "planning" method
        through "choose_action" method.
        """
        self.finish_rule = finish_rule

    def choose_action(self, _task, _value_function, state):
        """Utility method for calling "planning" method in the common format of
        this library. If you use this method, you must call "set_finish_rule"
        before calling this method.

        Raises:
            AssertionError: if "set_finish_rule" is not called
        """
        assert self.finish_rule is not None
        return self.planning(state, self.finish_rule)

    def planning(self, state, finish_rule):
        """Main logic of MCTS.

        - Algorithm -
        Initialize:
            R <- Build game tree which has only a root node which represents
                 passed state
        Repeat until satisfies finish condition of passed finish_rule:
            N <- descend R until not expanded node found.
                 (expanded = visited all child edge at least once)
            N' <- child node of new edge expanded at N
            R <- reward of simulation started from the state which N' represents
            Backpropagate R from N' to R
        return best action(edge of highest value) at R

        Args:
            state: start state to run MCTS procedure. It would be current state
                   of your agent in the task.
            finish_rule: child class of "kyoka.callback.FinishRule". This rule
                         is used to determine when to finish the MCTS iteration.
        Returns:
            action: best action at passed state calculated by MCTS algorithm.
        Raises:
            AssertionError: if passed state is terminal state, nothing to do.
        """
        assert not self.task.is_terminal_state(state)
        _log_start_msg(finish_rule)
        iteration_count = 0

        # TODO use last calculation result like AlphaGo
        root_node = self.generate_node_from_state(state)
        while not finish_rule.check_condition(iteration_count, self.task, None):
            finish_rule.before_update(iteration_count, self.task, None)

            selected_node = self._select(root_node)
            if self.task.is_terminal_state(selected_node.state):
                leaf_node = selected_node
                reward = self.task.calculate_reward(leaf_node.state)
            else:
                leaf_node = self._expand(selected_node)
                reward = self._playout(leaf_node)
            self._backpropagation(leaf_node, reward)

            finish_rule.after_update(iteration_count, self.task, None)
            iteration_count += 1

        self.last_calculated_tree = root_node
        _log_finish_msg(finish_rule, iteration_count)
        return root_node.greedy_edge.action

    def _select(self, root_node):
        """Find terminal or not expanded node"""
        target_node = root_node
        while target_node.is_expanded:
            target_node = target_node.select_best_edge().child_node
        return target_node

    def _expand(self, node):
        """Expand new edge from passed node"""
        assert node.has_unvisited_edge
        edge = node.select_unvisited_edge()
        edge.build_child(self.generate_node_from_state)
        return edge.child_node

    def _playout(self, leaf_node):
        """Run simulation from state of passed node and return reward"""
        reward = self.playout_policy(self.task, leaf_node)
        return reward

    def _backpropagation(self, leaf_node, reward):
        """Backpropagate passed reward from leaf_node to root node"""
        target = leaf_node
        assert target.parent_edge is not None
        while target.parent_edge:
            target.parent_edge.visit()
            target.parent_edge.update_by_new_reward(reward)
            target = target.parent_edge.parent_node

def random_playout(task, leaf_node, rand=random):
    """Default implmentation of playout policy of BaseMCTS.
    Simulation is played by the action choosed from possible actions
    at random.

    Args:
        task: run simulation by using this task
        leaf_node: start simulation from leaf_node.state
        rand: rand.choice is used to choose action
    Returns:
        reward: result of simulation
    """
    state = leaf_node.state
    while not task.is_terminal_state(state):
        actions = task.generate_possible_actions(state)
        action = random.choice(actions)
        state = task.transit_state(state, action)
    return task.calculate_reward(state)

def _log_start_msg(finish_rule):
    finish_rule.log(finish_rule.generate_start_message().replace("GPI", "MCTS"))

def _log_finish_msg(finish_rule, iteration_count):
    fin_msg= finish_rule.generate_finish_message(iteration_count)
    finish_rule.log(fin_msg.replace("GPI", "MCTS"))

class BaseNode(object):
    """Base class to build Node class of tree for MCTS

    This class is responsible for the logic of
    1. "how to build an Edge from the action"
    2. "how to choose the child node to descent in Select procedure of MCTS"

    The logic of 1. is abstracted as "generate_edge" method.
    The logic of 2. is implemented as "select_best_edge" method. Default
    implementation choose the edge which has highest value.

    Property:
        task: task passed from "BaseMCTS.generate_node_from_state"
        state: state of the node passed from "BaseMCTS.generate_node_from_state"
        parent_edge: reference of parent edge to access parent node
        child_edges: array of reference of edge to access child nodes
        greedy_edge: the edge which has highest average_reward
        terminal: boolean. True if state of this node is terminal.
        is_expanded: boolean. False if this node is possible to expand.
        has_unvisited_edge: boolean. True if one of edge.visit_count is 0.
        visit_count: sum of visit counts of child edges
    """

    def __init__(self, task, state):
        self.task = task
        self.state = state
        self.parent_edge = None
        self.child_edges = self.build_child_edges(task, state)
        self.terminal = task.is_terminal_state(state)

    def generate_edge(self, parent_node, action):
        """Define how to create Edge object from passed action

        If you have MyEdge class and use it for all child edges, code is simple

            def generate_edge(self, parent_node, action):
                return MyEdge(parent_node, action)

        Args:
            parent_node: parent node of the edge you will create
            action: you create the Edge object which represents this action
        Returns:
            edge: Edge object(child class of BaseEdge) which represents the
                  passed action
        """
        err_msg = build_not_implemented_msg(self, "generate_edge")
        raise NotImplementedError(err_msg)

    def select_best_edge(self):
        """Define how to choose next node to descend in SELECT procedure"""
        return sorted([edge for edge in self.child_edges], key=self._edge_sort_key)[-1]

    def _edge_sort_key(self, edge):
        return edge.calculate_value()

    def build_child_edges(self, task, state):
        actions = task.generate_possible_actions(state)
        return [self.generate_edge(self, action) for action in actions]

    def select_unvisited_edge(self):
        return [edge for edge in self.child_edges if not edge.has_child][0]

    @property
    def greedy_edge(self):
        best = max([edge.average_reward for edge in self.child_edges])
        best_edges = [edge for edge in self.child_edges if edge.average_reward==best]
        return best_edges[0]

    @property
    def is_expanded(self):
        """boolean. False if this node is possible to expand."""
        return not (self.terminal or self.has_unvisited_edge)

    @property
    def has_unvisited_edge(self):
        """boolean. True if one of edge.visit_count is 0."""
        return any([not edge.has_child for edge in self.child_edges])

    @property
    def visit_count(self):
        """sum of visit counts of child edges"""
        return sum([edge.visit_count for edge in self.child_edges])

class BaseEdge(object):
    """Base class to build Edge class of tree for MCTS

    This class is responsible for the logic of
    1. "holds average of rewards received in backpropagation"
    2. "how to calculate the value of edge for Select of MCTS"

    There are lots of methods to calculate value for selection 
    like UCT, UCT+, ... . So this logic is abstracted as
    "calculate_value" method.
    When you implement "calculate_value" by your own, we recommend you
    to refer implementation of UCTEdge class.

    Property:
        parent_node: reference to the parent node of the edge
        child_node: reference to the child node. None until
                    "build_child" is called.
        visit_count: the number of update in backpropagation process
        has_child: True if already "build_child" is called. This means
                   "this node is already selected at least once".
        average_reward: average of rewards received ever
    """

    def __init__(self, parent_node, action):
        self.action = action
        self.parent_node = parent_node
        self.child_node = None
        self.visit_count = 0
        self.average_reward = 0

    def calculate_value(self):
        """Define how to calculate the value of this Edge.
        This is called from select procedure of MCTS.

        Returns:
            edge_value: the value of edge used in selection procedure.
        """
        err_msg = build_not_implemented_msg(self, "calculate_value")
        raise NotImplementedError(err_msg)

    def build_child(self, state2node):
        child_state = self.parent_node.task.transit_state(self.parent_node.state, self.action)
        self.child_node = state2node(child_state)
        self.child_node.parent_edge = self

    def visit(self):
        self.visit_count += 1

    def update_by_new_reward(self, new_reward):
        self.average_reward = self._calc_average_in_incremental_way(
                self.average_reward, self.visit_count, new_reward)

    def _calc_average_in_incremental_way(self, old_value, visit_count, new_reward):
        assert visit_count != 0
        return old_value + 1.0 / visit_count * (new_reward - old_value)

    @property
    def has_child(self):
        """True if already "build_child" is called. This means
        "this node is already selected at least once".
        """
        return self.child_node is not None


class UCTNode(BaseNode):
    """Concrete implementation of BaseNode for UCT search.
    All of the logic of UCT search is implemented in UCT edge class
    """

    def generate_edge(self, parent_node, action):
        return UCTEdge(parent_node, action)

class UCTEdge(BaseEdge):
    """Concrete implementation of BaseEdge for UCT search.

    The value of edge for selection is calculated by following equation

    UCT = average(past_rewards) + 2 * C * sqrt( 2 * log(N) / n )
        where C = hyper parameter, n = visit count of edge,
              N = visit count of parent Node

    The default value of C is "1 / sqrt(2)". Because if reward is in range of
    [0,1], this value of UCT search satisfies Hoeffding's inequality.
    """

    def __init__(self, parent_node, action):
        super(UCTEdge, self).__init__(parent_node, action)
        self.C = 0.7071067811865475 # 1 / math.sqrt(2)

    def calculate_value(self):
        if self.visit_count == 0:
            explore_term = float('inf')
        else:
            explore_term = math.sqrt(2 * math.log(self.parent_node.visit_count) / self.visit_count)
        return self.average_reward + 2 * self.C * explore_term


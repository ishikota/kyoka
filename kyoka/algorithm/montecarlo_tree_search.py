import random
import math

from kyoka.utils import build_not_implemented_msg
from kyoka.task import BaseTask


class BaseMCTS(object):

    def __init__(self, task):
        self.task = task
        self.playout_policy = random_playout
        self.last_calculated_tree = None
        self.finish_rule = None

    def generate_node_from_state(self, state):
        err_msg = build_not_implemented_msg(self, "generate_node_from_state")
        raise NotImplementedError(err_msg)

    def set_playout_policy(self, policy):
        self.playout_policy = policy

    def set_finish_rule(self, finish_rule):
        self.finish_rule = finish_rule

    def choose_action(self, _task, _value_function, state):
        assert self.finish_rule is not None
        return self.planning(state, self.finish_rule)

    def planning(self, state, finish_rule):
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
        return root_node.select_best_edge().action

    def _select(self, root_node):
        target_node = root_node
        # FIXME move this logic to BaseNode class as expandable property
        while not self.task.is_terminal_state(target_node.state) and not target_node.has_unvisited_edge():
            target_node = target_node.select_best_edge().child_node
        return target_node

    def _expand(self, node):
        assert node.has_unvisited_edge()
        edge = node.select_unvisited_edge()
        edge.build_child(self.generate_node_from_state)
        return edge.child_node

    def _playout(self, leaf_node):
        reward = self.playout_policy(self.task, leaf_node)
        return reward

    def _backpropagation(self, leaf_node, reward):
        target = leaf_node
        assert target.parent_edge is not None
        while target.parent_edge:
            target.parent_edge.visit()
            target.parent_edge.update_value(reward)
            target = target.parent_edge.parent_node

def random_playout(task, leaf_node, rand=random):
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

    def __init__(self, task, state):
        self.task = task
        self.state = state
        self.parent_edge = None
        self.child_edges = self.build_child_edges(task, state)

    def generate_edge(self, parent_node, action):
        err_msg = build_not_implemented_msg(self, "generate_edge")
        raise NotImplementedError(err_msg)

    def select_best_edge(self):
        return sorted([edge for edge in self.child_edges], key=self._edge_sort_key)[-1]

    def _edge_sort_key(self, edge):
        return edge.calculate_value()

    def build_child_edges(self, task, state):
        actions = task.generate_possible_actions(state)
        return [self.generate_edge(self, action) for action in actions]

    def has_unvisited_edge(self):
        return any([not edge.has_child() for edge in self.child_edges])

    def select_unvisited_edge(self):
        return [edge for edge in self.child_edges if not edge.has_child()][0]

    def visit_count(self):
        return sum([edge.visit_count for edge in self.child_edges])

class BaseEdge(object):

    def __init__(self, parent_node, action):
        self.action = action
        self.parent_node = parent_node
        self.child_node = None
        self.visit_count = 0

    def update_value(self, new_reward):
        err_msg = build_not_implemented_msg(self, "update_value")
        raise NotImplementedError(err_msg)

    def calculate_value(self):
        err_msg = build_not_implemented_msg(self, "calculate_value")
        raise NotImplementedError(err_msg)

    def has_child(self):
        return self.child_node is not None

    def build_child(self, state2node):
        child_state = self.parent_node.task.transit_state(self.parent_node.state, self.action)
        self.child_node = state2node(child_state)
        self.child_node.parent_edge = self

    def visit(self):
        self.visit_count += 1

class UCTNode(BaseNode):

    def generate_edge(self, parent_node, action):
        return UCTEdge(parent_node, action)

class UCTEdge(BaseEdge):

    def __init__(self, parent_node, action):
        super(UCTEdge, self).__init__(parent_node, action)
        self.C = 1.4142135623730951  # 1.41... = math.sqrt(2)
        self.value = 0

    def update_value(self, new_reward):
        self.value = self._calc_average_in_incremental_way(self.value, self.visit_count, new_reward)

    def calculate_value(self):
        if self.visit_count == 0:
            explore_term = float('inf')
        else:
            explore_term = math.sqrt(math.log(self.parent_node.visit_count()) / self.visit_count)
        return self.value + self.C * explore_term

    def _calc_average_in_incremental_way(self, old_value, visit_count, new_reward):
        assert visit_count != 0
        return old_value + 1.0 / visit_count * (new_reward - old_value)


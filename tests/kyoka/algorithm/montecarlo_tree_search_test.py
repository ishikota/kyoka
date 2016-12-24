from mock import patch

from kyoka.task import BaseTask
from kyoka.algorithm.montecarlo_tree_search import BaseMCTS, BaseNode, BaseEdge,\
        UCTNode, UCTEdge, random_playout
from kyoka.callback import WatchIterationCount
from tests.base_unittest import BaseUnitTest


class MCTSTest(BaseUnitTest):

    def setUp(self):
        self.mcts = TestMCTS(TestTask())

    def test_update_edge_value(self):
        with self.assertRaises(NotImplementedError) as e:
            BaseMCTS(TestTask()).generate_node_from_state("dummy")
        self.include("generate_node_from_state", e.exception.message)

    def test_choose_action(self):
        self.mcts.set_playout_policy(self.mcts._mock_playout)
        self.mcts.set_finish_rule(WatchIterationCount(1))
        action = self.mcts.choose_action("dummy", "dummy", "A")
        self.eq(1, action)

    def test_planning(self):
        self.mcts.set_playout_policy(self.mcts._mock_playout)

        def edge_check(edge, value, visit_count):
            self.almosteq(value, edge.average_reward, 0.01)
            self.eq(visit_count, edge.visit_count)

        action = self.mcts.planning("A", WatchIterationCount(1, verbose=0))
        nodeA = self.mcts.last_calculated_tree
        self.eq(1, action)
        edge_check(nodeA.child_edges[0], 2, 1)

        action = self.mcts.planning("A", WatchIterationCount(2, verbose=0))
        nodeA = self.mcts.last_calculated_tree
        self.eq(1, action)
        edge_check(nodeA.child_edges[1], 1, 1)

        action = self.mcts.planning("A", WatchIterationCount(3, verbose=0))
        nodeB = self.mcts.last_calculated_tree.child_edges[0].child_node
        self.eq(1, action)
        edge_check(nodeB.child_edges[0], 0.5, 1)

        action = self.mcts.planning("A", WatchIterationCount(4, verbose=0))
        nodeA = self.mcts.last_calculated_tree
        self.eq(1, action)
        edge_check(nodeA.child_edges[1], 1, 2)

        action = self.mcts.planning("A", WatchIterationCount(5, verbose=0))
        nodeA = self.mcts.last_calculated_tree
        nodeB = nodeA.child_edges[0].child_node
        self.eq(1, action)
        edge_check(nodeA.child_edges[0], 1.33, 3)
        edge_check(nodeB.child_edges[1], 1.5, 1)

        action = self.mcts.planning("A", WatchIterationCount(6, verbose=0))
        nodeA = self.mcts.last_calculated_tree
        self.eq(1, action)
        edge_check(nodeA.child_edges[1], 1, 3)

        action = self.mcts.planning("A", WatchIterationCount(7, verbose=1))
        nodeA = self.mcts.last_calculated_tree
        nodeB = nodeA.child_edges[0].child_node
        nodeD = nodeB.child_edges[1].child_node
        self.eq(1, action)
        edge_check(nodeD.child_edges[0], 0.1, 1)
        edge_check(nodeB.child_edges[1], 0.8, 2)
        edge_check(nodeA.child_edges[0], 1.025, 4)

        action = self.mcts.planning("A", WatchIterationCount(8, verbose=0))
        nodeA = self.mcts.last_calculated_tree
        self.eq(1, action)
        edge_check(nodeA.child_edges[1], 1, 4)

        action = self.mcts.planning("A", WatchIterationCount(9, verbose=0))
        nodeA = self.mcts.last_calculated_tree
        nodeB = nodeA.child_edges[0].child_node
        self.eq(5, action)
        edge_check(nodeB.child_edges[0], 0.5, 2)
        edge_check(nodeA.child_edges[0], 0.919, 5)

        action = self.mcts.planning("A", WatchIterationCount(10, verbose=0))
        nodeA = self.mcts.last_calculated_tree
        self.eq(5, action)
        edge_check(nodeA.child_edges[1], 1, 5)

        action = self.mcts.planning("A", WatchIterationCount(11, verbose=0))
        nodeA = self.mcts.last_calculated_tree
        self.eq(5, action)
        edge_check(nodeA.child_edges[1], 1, 6)

        action = self.mcts.planning("A", WatchIterationCount(12, verbose=0))
        nodeA = self.mcts.last_calculated_tree
        nodeB = nodeA.child_edges[0].child_node
        nodeD = nodeB.child_edges[1].child_node
        self.eq(5, action)
        edge_check(nodeD.child_edges[0], 0.1, 2)
        edge_check(nodeB.child_edges[1], 0.566, 3)
        edge_check(nodeA.child_edges[0], 0.78, 6)

        action = self.mcts.planning("A", WatchIterationCount(13, verbose=0))
        nodeA = self.mcts.last_calculated_tree
        self.eq(5, action)
        edge_check(nodeA.child_edges[1], 1, 7)

        self.almosteq(2.94, nodeA.child_edges[0].calculate_value(), 0.01)
        self.almosteq(2.85, nodeA.child_edges[1].calculate_value(), 0.01)

    def test_select(self):
        root = self.mcts.generate_node_from_state("A")
        self.eq("A", self.mcts._select(root).state)
        root.child_edges[0].build_child(self.mcts.generate_node_from_state)
        self.eq("A", self.mcts._select(root).state)
        root.child_edges[1].build_child(self.mcts.generate_node_from_state)

        root.child_edges[1].average_reward = 5
        self.eq("F", self.mcts._select(root).state)

        root.child_edges[0].average_reward = 10
        self.eq("B", self.mcts._select(root).state)

        nodeB = root.child_edges[0].child_node
        nodeB.child_edges[0].build_child(self.mcts.generate_node_from_state)
        self.eq("B", self.mcts._select(root).state)
        nodeB.child_edges[1].build_child(self.mcts.generate_node_from_state)
        nodeB.child_edges[1].average_reward = 10
        self.eq("D", self.mcts._select(root).state)

        nodeD = nodeB.child_edges[1].child_node
        nodeD.child_edges[0].build_child(self.mcts.generate_node_from_state)
        self.eq("E", self.mcts._select(root).state)

        nodeB.child_edges[0].average_reward = 20
        self.eq("C", self.mcts._select(root).state)

    def test_expand(self):
        root = self.mcts.generate_node_from_state("A")
        self.assertIsNone(root.child_edges[0].child_node)
        self.assertIsNone(root.child_edges[1].child_node)
        self.mcts._expand(root)
        self.eq("B", root.child_edges[0].child_node.state)
        self.mcts._expand(root)
        self.eq("F", root.child_edges[1].child_node.state)
        with self.assertRaises(AssertionError) as e:
            self.mcts._expand(root)

    def test_playout(self):
        root = self.mcts.generate_node_from_state("A")
        self.mcts._expand(root)
        self.mcts._expand(root)
        nodeF = root.child_edges[1].child_node
        with patch("random.choice", side_effect=lambda ary: ary[0]):
            self.eq(0.5, self.mcts._playout(root))
        self.eq(1, self.mcts._playout(nodeF))

    def test_backpropagation(self):
        nodeA = self.mcts.generate_node_from_state("A")
        nodeA.child_edges[0].build_child(self.mcts.generate_node_from_state)
        nodeA.child_edges[1].build_child(self.mcts.generate_node_from_state)
        nodeB = nodeA.child_edges[0].child_node
        nodeB.child_edges[0].build_child(self.mcts.generate_node_from_state)
        nodeB.child_edges[1].build_child(self.mcts.generate_node_from_state)
        nodeC = nodeB.child_edges[0].child_node
        nodeD = nodeB.child_edges[1].child_node
        nodeD.child_edges[0].build_child(self.mcts.generate_node_from_state)
        nodeE = nodeD.child_edges[0].child_node
        nodeF = nodeA.child_edges[1].child_node
        edge1 = nodeA.child_edges[0]
        edge2 = nodeB.child_edges[0]
        edge3 = nodeB.child_edges[1]
        edge4 = nodeD.child_edges[0]
        edge5 = nodeA.child_edges[1]

        def subtest(edge_values, visit_counts):
            self.eq(edge1.average_reward, edge_values[0])
            self.eq(edge1.visit_count, visit_counts[0])
            self.eq(edge2.average_reward, edge_values[1])
            self.eq(edge2.visit_count, visit_counts[1])
            self.eq(edge3.average_reward, edge_values[2])
            self.eq(edge3.visit_count, visit_counts[2])
            self.eq(edge4.average_reward, edge_values[3])
            self.eq(edge4.visit_count, visit_counts[3])
            self.eq(edge5.average_reward, edge_values[4])
            self.eq(edge5.visit_count, visit_counts[4])

        self.mcts._backpropagation(nodeB, 1)
        subtest([1, 0, 0, 0, 0], [1, 0, 0, 0, 0])
        self.mcts._backpropagation(nodeC, 3)
        subtest([2, 3, 0, 0, 0], [2, 1, 0, 0, 0])
        self.mcts._backpropagation(nodeD, 5)
        subtest([3, 3, 5, 0, 0], [3, 1, 1, 0, 0])
        self.mcts._backpropagation(nodeE, -1)
        subtest([2, 3, 2, -1, 0], [4, 1, 2, 1, 0])
        self.mcts._backpropagation(nodeF, 5)
        subtest([2, 3, 2, -1, 5], [4, 1, 2, 1, 1])

    def test_random_playout(self):
        root = self.mcts.generate_node_from_state("A")
        with patch("random.choice", side_effect=lambda ary: ary[0]):
            self.eq(0.5, random_playout(TestTask(), root))
        with patch("random.choice", side_effect=lambda ary: ary[1]):
            self.eq(1, random_playout(TestTask(), root))

class BaseNodeTest(BaseUnitTest):

    def test_generate_edge(self):
        with self.assertRaises(NotImplementedError) as e:
            BaseNode(TestTask(), "A")
        self.include("generate_edge", e.exception.message)

    def test_select_best_edge(self):
       node = TestNode(TestTask(), "A")
       node.child_edges[0].visit()
       node.child_edges[0].update_by_new_reward(1)
       self.eq(1, node.select_best_edge().action)

    def test_select_greedy_edge(self):
       node = TestNode(TestTask(), "A")
       node.child_edges[0].visit()
       node.child_edges[0].update_by_new_reward(1)
       node.child_edges[1].visit()
       node.child_edges[1].update_by_new_reward(1.1)
       node.child_edges[1].visit()
       node.child_edges[1].update_by_new_reward(1.1)
       self.eq(node.child_edges[0], node.select_best_edge())
       self.eq(node.child_edges[1], node.greedy_edge)

    def test_build_child_edges(self):
       node = TestNode(TestTask(), "A")
       self.eq(2, len(node.child_edges))
       self.eq(1, node.child_edges[0].action)
       self.eq(node, node.child_edges[0].parent_node)
       self.eq(5, node.child_edges[1].action)
       self.eq(node, node.child_edges[1].parent_node)

    def test_has_unvisited_edge(self):
       node = TestNode(TestTask(), "A")
       state2node = lambda state: TestNode(TestTask(), state)
       self.true(node.has_unvisited_edge)
       node.child_edges[0].build_child(state2node)
       self.true(node.has_unvisited_edge)
       node.child_edges[1].build_child(state2node)
       self.false(node.has_unvisited_edge)

    def test_select_unvisited_edge(self):
       node = TestNode(TestTask(), "A")
       state2node = lambda state: TestNode(TestTask(), state)
       self.false(node.child_edges[0].has_child)
       node.select_unvisited_edge().build_child(state2node)
       self.true(node.child_edges[0].has_child)
       self.false(node.child_edges[1].has_child)
       node.select_unvisited_edge().build_child(state2node)
       self.true(node.child_edges[1].has_child)

    def test_visit_count(self):
       node = TestNode(TestTask(), "A")
       self.eq(0, node.visit_count)
       node.child_edges[0].visit()
       node.child_edges[0].visit()
       node.child_edges[1].visit()
       self.eq(3, node.visit_count)

class BaseEdgeTest(BaseUnitTest):

    def setUp(self):
        self.edge = BaseEdge(TestNode(TestTask(), "A"), 1)

    def test_update_by_new_reward(self):
        self.edge.visit()
        self.edge.update_by_new_reward(5)
        self.eq(5, self.edge.average_reward)
        self.edge.visit()
        self.edge.update_by_new_reward(1)
        self.eq(3, self.edge.average_reward)

    def test_calculate_value(self):
        with self.assertRaises(NotImplementedError) as e:
            self.edge.calculate_value()
        self.include("calculate_value", e.exception.message)

    def test_build_child_and_had_child(self):
        edge = BaseEdge(TestNode(TestTask(), "A"), 1)
        self.false(edge.has_child)
        edge.build_child(lambda state: TestNode(TestTask(), state))
        self.eq("B", edge.child_node.state)
        self.true(edge.has_child)

    def test_visit(self):
        self.eq(0, self.edge.visit_count)
        self.edge.visit()
        self.eq(1, self.edge.visit_count)

class UCTEdgeNodeTest(BaseUnitTest):

    def setUp(self):
        self.nodeA = UCTNode(TestTask(), "A")
        self.edge = self.nodeA.child_edges[0]

    def test_calculate_value(self):
        self.edge.visit()
        self.edge.update_by_new_reward(0)
        self.almosteq(0, self.edge.calculate_value(), 0.0001)
        self.nodeA.child_edges[1].visit()
        self.almosteq(1.6651092223153954, self.edge.calculate_value(), 0.0001)
        self.edge.visit()
        self.edge.update_by_new_reward(1)
        self.almosteq(1.982303807367511, self.edge.calculate_value(), 0.0001)

class TestTask(BaseTask):

    def is_terminal_state(self, state):
        return state in ["C", "E", "F"]

    def transit_state(self, state, action):
        return { "A":{1:"B", 5:"F"}, "B":{2:"C", 3:"D"}, "D":{4:"E"} }[state][action]

    def generate_possible_actions(self, state):
        return { "A":[1,5], "B":[2,3], "C":[], "D":[4], "E":[], "F":[] }[state]

    def calculate_reward(self, state):
        return { "C": 0.5, "E": 0.1, "F": 1 }[state]

class TestMCTS(BaseMCTS):

    def generate_node_from_state(self, state):
        return TestNode(self.task, state)

    def _mock_playout(self, task, leaf_node):
        state = leaf_node.state
        if task.is_terminal_state(state):
            return task.calculate_reward(state)
        else:
            return { "B": 2, "D":1.5 }[state]

class TestNode(BaseNode):

    def generate_edge(self, parent_node, action):
        return TestEdge(parent_node, action)

class TestEdge(BaseEdge):

    def calculate_value(self):
        if self.visit_count == 0:
            explore_term = 0
        else:
            explore_term = 1.0 * self.parent_node.visit_count / self.visit_count
        return self.average_reward + explore_term


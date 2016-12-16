# MonteCarloTreeSearch
Monte Carlo Tree Search (MCTS) is one of the famous **heuristic search** method. MCTS builds search tree to find most promising action by using results of random simulation.  

Unlike other reinforcement algorithms (ex. MonteCarlo, QLearning, ...), MCTS has no training phase.  
MCTS finds most promising action every time when it receives new state to choose action.  

For more detail explanation see [A Survey of Monte Carlo Tree Search Methods](http://ieeexplore.ieee.org/document/6145622/).  
(We implemented MCTS based on this paper.)

## Algorithm
The search tree of MCTS represents search space of reinforcement learning task.  
Each node represents some state *S* and its child edge represents possible actions at state *S*.  

First we pass current state of task to MCTS. Then MCTS creates search tree which has only root node representing current state.
After that MCTS starts to build search tree by iterating following 4 steps as long as possible.

1. SELECT : select a node which has not expanded edge
2. EXPAND : add child node of selected node on search tree
3. PLAYOUT : run random simulation from state which expanded node representing
4. BACKPROPAGATION : backpropagate reward of simulation from expanded node to root node

```
Initialize:
    R <- Build game tree which has only a root node which represents
         current state (where we want to find best action)
Repeat until computational budget runs out:
    N  <- find not expanded node by descending R from root node
         (expanded = visited all child node at least once)
    N' <- child node of N which is not visited yet.
          And add N' on search tree.
    R  <- reward of simulation started from the state which N' represents
    Backpropagate R from N' to R
return best action(edge of highest value) at R
```

## Tutorial
Now we will introduce how to use MCTS with example task `TickTackToeTask`.  

Before starting implementation, we need to decide the algorithm for *SELECT* step (how to choose the node to expand).
In this tutorial we adapt famous algorithm *UCT search* (Most of the case this choice would be good).

*UCT search* calculate the value of edge by following equation.  
```
edge_value = average_reard + 2 * C * sqrt( 2 * log(N) / n )
```
where

- `average_reward` is the average of reward received through *BACKPROPAGATION* step.
- `C` is the hyper parameter to balance explore and exploitation
- `N` total visit count of parent node of target edge
- `n` total visit count of child node of target edge

And descend to the edge of child node which has maximum value.  

This algorithm is already implemented as `UCTNode`, `UCTEdge` classes. So we will use it.

### Creating custom node
*tick-tack-toe* is two-player zero-sum game. So the best action of opponent player is the worst action of my player.  
To integrate this idea with UCT search, we create `MaxNode` and `MinNode`.

```python
from kyoka.algorithm.montecarlo_tree_search import UCTNode

# descend the tree to the child edge which has maximum value. (choose best action for me)
# This node should represents the state of my turn.
class MaxNode(BaseNode):

    def select_best_edge(self):
        sort_key = lambda edge: edge.calculate_value()
        max_val_edge = sorted([edge for edge in self.child_edges], key=sort_key)[-1]
        return max_val_edge

# descend the tree to the child edge which has minimum value. (choose worst action for me)
# This node should represents the state of opponent turn.
class MinNode(BaseNode):

    def select_best_edge(self):
        sort_key = lambda edge: edge.calculate_value()
        min_val_edge = sorted([edge for edge in self.child_edges], key=sort_key)[0]
        return min_val_edge
```

After finished to create `Node` class, we need to tell MCTS how to build our search tree.  
We will do this by implementing `BaseMCTS.generate_node_from_state` method.

```
from kyoka.algorithm.montecarlo_tree_search import BaseMCTS

class MyMCTS(BaseMCTS):

    def generate_node_from_state(self, state):
        if self.next_player_is_me(state):
            return MaxNode(state)
        else:
            return MinNode(state)

    def next_player_is_me(self, state):
        return # TODO judge passed state is my turn or not by some logic
```

Ok, we prepared everything.  
Below code runs 5000 iteration of MCTS and returns most promising action at initial state.

```python
from kyoka.callback import WatchIterationCount

NB_SIMULATION = 5000
finish_rule = WatchIterationCount(SIMULATION_NUM)
task = TickTackToeTask()
algorithm = MyMCTS(TickTackToeTask)

state = task.generate_initial_state()
best_action = algorithm.planning(state, finish_rule)

# you can also call planning method through "choose_action(task, value_functoin, state)" interface
algorithm.set_finish_rule(finish_rule)
best_action = algorithm.choose_action("dummy", "dummy", state)
```

## Advanced
If you want to run simulation in not random way, you can customize simulation logic.  
Simulation is executed by calling method from `BaseMCTS.playout_policy` property.  
So you can create simulation method and set it by calling `BaseMCTS.set_playout_policy` method.  

The interface of simulation method is

- receives *task* and *leaf_node* as argument
- returns reward of simulation

The default implementation `random_playout` is implemented like this.
```python
import random

def random_playout(task, leaf_node):
    state = leaf_node.state
    while not task.is_terminal_state(state):
        actions = task.generate_possible_actions(state)
        action = random.choice(actions)
        state = task.transit_state(state, action)
    return task.calculate_reward(state)

mcts.set_playout_policy(random_playout)
```



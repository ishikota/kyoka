# kyoka - Reinforcement Learning framework

### What is Reinforcement Learning
>Reinforcement learning is an area of machine learning inspired by behaviorist psychology, concerned with how software agents ought to take actions in an environment so as to maximize some notion of cumulative reward. (wikipedia)

In reinforcement learning, the player to learn how to get good result in some task is called as **agent**.  
Agent learns which action is good or bad at each situation through large number of simulation.  
(The essential factor to characterize reinforcement learning is **learning from trial-and-error**.)

## Why *kyoka* is creted
The steps to solve your learning problem (ex. playing Go) by reinforcement learning algorithms would be  

1. **Define your learning problem** in Reinforcement Learning format.  
2. Select learning algorithm(ex. QLearning) and **implement it for your learning problem**.  

We have a lots of things to do before start learning.  
This library is created to ease implementing these steps.  

Sorry, I talked too much. Let's see the code with simple example !!

## Hello Reinforcement Learning
We will find the shortest path to escape from below maze by `QLearning`.
```bash
S: start, G: goal, X: wall

-------XG
--X----X-
S-X----X-
--X------
-----X---
---------
```

### Step1. Define Maze Task
First we define our learning problem as reinforcement learning task.
*kyoka* provides `kyoka.task.BaseTask` template class. This class has 5 abstracted methods you need to implement.

1. `gegenerate_inital_state` : define start state of our problem
2. `is_terminal_state` : define when is the finish of our problem
3. `transit_state` : define the rule of state transition in our problem
4. `generate_possible_actions` : define what action is possible in each state
5. `calculate_reward` : define how good each state is

Here is the `MazeTask` class which represents our learning problem.

```python
from kyoka.task import BaseTask

class MazeTask(BaseTask):

    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_RIGHT = 2
    ACTION_LEFT = 3

    # We use current position of the agent in the maze as "state".
    # So we return start position of the maze (row=2, col=0).
    def generate_initial_state(self):
        return (2, 0)

    # The position of the goal is (row=0, column=8).
    def is_terminal_state(self):
        return (0, 8) == state

    # We can always move towards 4 directions.
    def generate_possible_actions(self, state):
        return [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_RIGHT, self.ACTION_LEFT]

    # Agent can get reward +1 only when he reaches to the goal.
    def calculate_reward(self, state):
        return 1 if self.is_terminal_state(state) else 0

    # Returns next state after moved toward direction of passed action.
    # If destination is out of the maze or block cell, do not move.
    def transit_state(self, state, action):
        row, col = state
        wall_position = [(1,2), (2,2), (3,2), (4,5), (0,7), (1,7), (2,7)]
        height, width = 6, 9
        if self.ACTION_UP == action:
            row = max(0, row-1)
        elif self.ACTION_DOWN == action:
            row = min(height-1, row+1)
        elif self.ACTION_RIGHT == action:
            col= min(width-1, col+1)
        elif self.ACTION_LEFT == action:
            col = max(0, col-1)
        if (row, col) not in wall_position:
            return (row, col)
        else:
            return state  # Stay current position if destination is not a path.

```

### Step2. Setup QLearning for MazeTask
Next we implement **value function** of our `MazeTask` for `QLearning`.  

**value function** is the function which receives **state-action** pair and estimates **how good** for the agent to take the action at the state.
So value function would work like this

```python
value_of_action = value_function.predict_value(state=(1, 5), action=ACTION_UP)
# value_of_action should be 1 because (1,6) is the goal of the maze.
```

The most important part of reinforcement learning is to **learn correct value function** of the task.

Each algorithm in this library has different base class of value function  
(ex. `QLearningTabularActionValueFunction`, `DeepQLearningApproxActionValueFunction`).  
Now we need to implement abstracted method of `QLearningTabularActionValueFunction`.

Here is the `MazeTabularValueFunction` class for `QLearning`.
```python
class MazeTabularValueFunction(QLearningTabularActionValueFunction):

    # We use table(array) to store the value of state-action pair.
    # Ex. the value of action=ACTION_RIGHT at state=(0,3) is stored in table[0][3][2].
    def generate_initial_table(self):
        maze_width, maze_height, action_num = 6, 9, 4
        return [[[0 for a in range(action_num)] for j in range(width)] for i in range(height)]

    # Define how to fetch value from the table which
    # initialized by "generate_initial_table" method.
    def fetch_value_from_table(self, table, state, action):
        row, col = state
        return table[row][col][action]

    # Define how to update the value of table.
    def insert_value_into_table(self, table, state, action, new_value):
        row, col = state
        table[row][col][action] = new_value
```

### Final Step. Run `QLearning` and see its result
Ok, we prepared everything. Next code starts the learning.
```python
task = MazeTask()
policy = EpsilonGreedyPolicy(eps=0.1)
value_function = MazeTabularValueFunction()
algorithm = QLearning()
algorithm.setup(task, policy, value_function)  # setup before calling "run_gpi"
algorithm.run_gpi(nb_iteration=100)  # starts the learning
```

That's all !! Now `value_function` stores how good each action is. Let's visualize what agent learned.  
(We prepared helper method `examples.maze.helper.visualize_policy`.)

```python
>>> print visualize_policy(task, value_function)

     -------XG
     --X-v-vX^
S -> v-X-vvvX^
     vvX>>>>>^
     >>>>^-^^^
     ->^<^----
```

Great!! Agent found the shortest path to the goal. (14 step is the minimum step to the goal !!)

## Installation
You can use pip like this.
```
pip install kyoka
```

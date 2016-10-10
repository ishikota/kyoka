# kyoka : Simple Reinforcement Learning Library
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/ishikota/kyoka/blob/master/LICENSE.md)
[![PyPI](https://img.shields.io/pypi/v/kyoka.svg?maxAge=2592000)]([![PyPI](https://img.shields.io/pypi/v/nine.svg?maxAge=2592000)](https://github.com/ishikota/kyoka))
## Implemented algorithmes
- MonteCarlo
- Sarsa
- QLearning
- SarsaLambda
- QLambda

Algorithms are implemented based on the book [Sutton & Barto Book: Reinforcement Learning: An Introduction](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html)

# Getting Started
## Motivation
RL(Reinforcement Learning) algorithms  learns which action is good or bad through **trial-and-error**.  
So what we need to do is **making our learning task in RL format**.

This library provides two template classes to make your task in RL format.
- `BaseDomain` class which represents our learning task
- `ValueFunction` class which RL algorithm uses to save trial-and-error result

So let's see how to use these template classes through simple example.

## Example. Find best policy to escape from the maze
Here we will find the best policy to escape from the below maze by RL algorithm.
```
S: start, G: goal, X: wall

-------XG
--X----X-
S-X----X-
--X------
-----X---
---------
```

### Step1. Create MazeDomain class
`BaseDomain` class requires you to implement 5 methods
- `generate_initial_state()`
  - returns initial state that RL algorithms starts simulation from.
- `generate_possible_actions(state)`
  - returns valid actions in passed state. RL algorithms choose next action from these actions.
- `transit_state(state, action)`
  - returns next state after applied the passed action on the passed state.
- `calculate_reward(state)`
  - returns how good the passed state is.
- `is_terminal_state(state)`
  - returns if passed state is terminal state or not.
  
```
class MazeDomain(BaseDomain):

  ACTION_UP = 0
  ACTION_DOWN = 1
  ACTION_RIGHT = 2
  ACTION_LEFT = 3

  // we use current position of the maze as "state". So here we return start position of the maze.
  def generate_initial_state(self):
    return (0, 0)

  // the position of the goal is (row=3, column=2)
  def is_terminal_state(self, state):
    return (3, 2) == state
  
  // we can always move to 4 directions.
  def generate_possible_actions(self, state):
    return [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_RIGHT, self.ACTION_LEFT]
  
  // RL algorithm can get reward only when he reaches to the goal.
  def calculate_reward(self, state):
    return 1 if self.is_terminal_state(state) else 0

  def transit_state(self, state, action):
    row, col = state
    height, width = 6, 9
    if action == self.UP:
      row = max(0, row-1)
    elif action == self.DOWN:
      row = min(height-1, row+1)
    elif action == self.RIGHT:
      col= min(width-1, col+1)
    elif action == self.LEFT:
      col = max(0, col-1)
    if 'X' != self.maze[row][col]:
      return (row, col)
    else:
      return state // If destination is the wall or edge of the maze then position does not change.
    
```

Ok! next is `ValueFunction`!!

### Step2. Create MazeActionValueFunction class
`BaseActionValueFunction` class requires you to implement 2 methods.
- `calculate_value(state, action)`
  - fetch current value of state and action pair.
- `update_value(state, action, new_value)`
  - update Q-value of passed state and action by passed value.

The state space of this example is very small (state space = |state| x |action| = 12 x 4 = 48).  
So we prepare the table (2-dimentional array) and save value on it.

```
class MazeActionValueFunction(BaseActionValueFunction):

  // call this method before start learning
  def setUp(self):
    maze_cell_num, action_num = 48, 4
    self.table = [[0 for j in range(action_num)] for i in range(maze_cell_num)]

  // just take value from the table
  def calculate_value(self, state, action):
    return self.table[state][action]
  
  // just insert value into the table
  def update_value(self, state, action, new_value):
    self.table[state][action] = new_value
```

#### hint: Deep Reinforcement Learning
If state space is too learge, you can use neural net as value function like [DQN](https://arxiv.org/pdf/1312.5602.pdf).  

If you are interested in it, you can checkout [`BaseKerasValueFunction` ](https://github.com/ishikota/kyoka/blob/master/kyoka/value_function/base_keras_action_value_function.py)  
(`BaseKerasValueFunction` internally  uses [keras](https://github.com/fchollet/keras) library to approximate value function by neuralnet. )

The sample implementation of `BaseKerasValueFunction` for maze domain is [here (MazeKerasValueFunction)](https://github.com/ishikota/kyoka/blob/master/sample/maze/maze_keras_value_function.py).

### Step3. Running RL algorithm and see its result
OK, here we apply `QLearning` on our *maze*  RL task.

```
rl_algo = QLearning(alpha=0.1, gamma=0.7) // You can replace RL algorithm like "rl_algo = Sarsa(alpha=0.1, gamma=0.7)"
domain = MazeDomain()
policy = EpsilonGreedyPolicy(epsilon=0.1)
value_function = MazeValueFunction()
value_function.setUp()
finish_rule = WatchIterationCount(target_count=50)  // finish learning iteration after 50-th GPI iteration
rl_algo.GPI(domain, policy, value_function, finish_rule)
```

That's all !! Let's visualize value function which QLearning learned.
```
      -------XG
      --X-v-vX^
 S -> v-X-vvvX^
      vvX>>>>>^
      >>>>^-^^^
      ->^<^----
```

Looks good!! QLearning found the policy which leads us to goal in 14 steps. (14 step is minimum step to the goal !!)

## Sample code
In sample directory, we prepare complete sample code as jupyter notebook and script.
You can also checkout another RL task example *tick-tack-toe* .
- [sample: Learning how to escape from maze by RL](https://github.com/ishikota/kyoka/tree/master/sample/maze)
- [smaple: Learning tick-tack-toe by RL](https://github.com/ishikota/kyoka/tree/master/sample/ticktacktoe)

# Installation
under construction... :bow:

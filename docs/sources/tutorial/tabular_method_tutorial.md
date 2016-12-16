# Tabular Reinforcement Learning Problem
In this tutorial, we will solve the problem called **tabular reinforcement learning problem**.  

The keyword **tabular** means **state-action space of the problem is small enough to fit in array or table**.  
Most of reinforcement learning methods have good convergence property on *tabular reinforcement learning problem*.  
So first we will approach this type of problem.

## blocking maze problem
We will find the shortest path of *blocking maze*.  
*blocking maze* transforms its structure during training like below.  

```bash
  Before         After
 --------G     --------G
 ---------     ---------
 ---------  => ---------
 XXXXXXXX-     -XXXXXXXX
 ---------     ---------
 ---S-----     ---S-----
# 10 step       16 step  <= minimum step
```
The best path in first structure is blocked by transformation.  
So agent needs to realize the transformation and re-learn shortest path.

## Implementation
### Define blocking maze task
First we define our *blocking maze problem* as reinforcement learning task.
What we need to do is implementing 5 abstracted methods of `BaseTask` class.
So our `BlockingMazeTask` would be like this.

```python
from kyoka.task import BaseTask

class BlockingMazeTask(BaseTask):

    UP, DOWN, RIGHT, LEFT = 0, 1, 2, 3
    START, GOAL = (5, 3), (0, 8)
    BEFORE_MAZE = [
            ['-','-','-','-','-','-','-','-','G'],
            ['-','-','-','-','-','-','-','-','-'],
            ['-','-','-','-','-','-','-','-','-'],
            ['X','X','X','X','X','X','X','X','-'],
            ['-','-','-','-','-','-','-','-','-'],
            ['-','-','-','S','-','-','-','-','-']
    ]
    AFTER_MAZE = [
            ['-','-','-','-','-','-','-','-','G'],
            ['-','-','-','-','-','-','-','-','-'],
            ['-','-','-','-','-','-','-','-','-'],
            ['-','X','X','X','X','X','X','X','X'],
            ['-','-','-','-','-','-','-','-','-'],
            ['-','-','-','S','-','-','-','-','-']
    ]

    def __init__(self):
        self.maze = self.BEFORE_MAZE

    def generate_initial_state(self):
        return self.START

    def is_terminal_state(self, state):
        return self.GOAL == state

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
            return state

    def generate_possible_actions(self, state):
        return [self.UP, self.DOWN, self.RIGHT, self.LEFT]

    def calculate_reward(self, state):
        row, col = state
        return 1 if 'G' == self.maze[row][col] else 0
```

This code does not imeplment transformation feature yet.  
We implement the transformation feature by using `kyoka.callback` module later.

### Setup algorithm for blocking maze problem
Next we create **value function** of our task.
>value function is the function which receives state-action pair and estimates how good for the agent to take the action at the state.

Before start implementation, let's think about the **size of state-action space** of our blocking maze problem.  
In our `BlockingMazeTask`, state and action is defined like this.

- state  = position of agent in the maze 
- action = the direction to move (UP or DOWN or RIGHT or LEFT)

The number of possible state is `6*9=54`(our maze shape is 6x9) and we can move `4` direction in every state.
So the **size of state-action space** is `54*4=216`.  
This indicates that our learning problem is *small enough to fit in array or table*.  
So we can say that blocking maze problem is *tabular reinforcement learning problem*.

Ok, let's resume implementation.
Here we use `Sarsa` method to find shortest path.  
What we need to do is implementing abstract methods of `SarsaTabularActionValueFunction` like this.

```python
class MazeTabularValueFunction(SarsaTabularActionValueFunction):

    def generate_initial_table(self):
        maze_width, maze_height, action_num = 6, 9, 4
        return [[[0 for a in range(action_num)] for j in range(width)] for i in range(height)]

    def fetch_value_from_table(self, table, state, action):
        row, col = state
        return table[row][col][action]

    def insert_value_into_table(self, table, state, action, new_value):
        row, col = state
        table[row][col][action] = new_value
```

### Implement maze transformation feature
We implement maze transformation feature by using `keras.callback` module.
This module provides the callback methods to interact with our task and value function under training.  

Base class of all callback is `keras.callback.BaseCallback`. This class has 4 callback methods.

- `before_gpi_start(task, value_function)`
- `before_update(iteration_count, task, value_function)`
- `after_update(iteration_count, task, value_function)`
- `after_gpi_finish(task, value_function)`

Here we create the `MazeTransformationCallback` which interacts with `BlockingMazeTask` after 50 iteration of training and switch the shape of maze. The code is like this.
```python
from kyoka.callback import BaseCallback

class MazeTransformationCallback(BaseCallback):

    def after_update(self, iteration_count, task, value_function):
        if 50 == iteration_count:
            task.maze = BlockingMazeTask.AFTER_MAZE
            # we recommend you to use "self.log(message)" instead of "print" method in callback.
            self.log("Maze transformed after %d iteration." % iteration_count)

```

### Watch performance of agent during training
Before start training, we implement one more important callback `MazePerformanceWatcher`.  
This callback logs performance of agent in each iteration of training.  
(In our case, performance means *how many step does agent takes to the goal*.)

With `task` and `value_function`, we can test performance of agent like this.
```python
from kyoka.policy import choose_best_action

MAX_STEP = 10000
def solve_maze(task, value_function):
    step_count = 0
    state = task.generate_initial_state()
    while not task.is_terminal_state():
        action = choose_best_action(task, value_function, state)
        state = task.transit_state(state, action)
        step_count += 1
        if step_count >= MAX_STEP:  # agent may never reaches to the goal
            break
    return step_count

```

We create `MazePerformanceWatcher` by using `kyoka.callback.BasePerformanceWatcher` callback.  
The code would be like this.

```python
class MazePerformanceWatcher(BasePerformanceWatcher):

    def define_performance_test_interval(self):
        return 1  # this means "run_performance_test" is called in every "after_update" callback

    # Do performance test and return its result.
    # Result is passed to "define_log_message" method as "test_result" argument.
    def run_performance_test(self, task, value_function):
        step_to_goal = solve_maze(task, value_function)
        return step_to_goal

    # The message returned here is logged after every "run_performance_test" is called.
    def define_log_message(self, iteration_count, task, value_function, test_result):
        step_to_goal = test_result
        return "Step = %d (iteration=%d)" % (step_to_goal,iteration_count)

```

## Solve blocking maze
Ok, we prepared everything to start training.  Let's start !!

```python
from kyoka.algorithm.sarsa import Sarsa
from kyoka.policy import EpsilonGreedyPolicy

task = BlockingMazeTask()
policy = EpsilonGreedyPolicy(eps=0.1)
value_function = MazeTabularValueFunction()
algorithm = Sarsa()
algorithm.setup(task, policy, value_function)

callbacks = [MazeTransformationCallback(), MazePerformanceWatcher()]
algorithm.run_gpi(nb_iteration=100, callbacks=callbacks)
```

Training logs would be output on console like this

```
[Progress] Start GPI iteration for 100 times
[Progress] Finished 1 / 100 iterations (0.0s)
[MazePerformanceWatcher] Step = 223 (nb_iteration=1)
[Progress] Finished 2 / 100 iterations (0.0s)
[MazePerformanceWatcher] Step = 507 (nb_iteration=2)
[Progress] Finished 3 / 100 iterations (0.0s)
[MazePerformanceWatcher] Step = 220 (nb_iteration=3)
[Progress] Finished 4 / 100 iterations (0.0s)
[MazePerformanceWatcher] Step = 142 (nb_iteration=4)
[Progress] Finished 5 / 100 iterations (0.0s)
[MazePerformanceWatcher] Step = 572 (nb_iteration=5)
[Progress] Finished 6 / 100 iterations (0.0s)
[MazePerformanceWatcher] Step = 781 (nb_iteration=6)
[Progress] Finished 7 / 100 iterations (0.0s)
[MazePerformanceWatcher] Step = 18 (nb_iteration=7)
...
[Progress] Finished 50 / 100 iterations (0.0s)
[MazePerformanceWatcher] Step = 10 (nb_iteration=50)
[MazeTransformationCallback] Maze transformed after 50 iteration
[Progress] Finished 51 / 100 iterations (1.6s)
[MazePerformanceWatcher] Step = 10000 (nb_iteration=51)
[Progress] Finished 52 / 100 iterations (1.4s)
[MazePerformanceWatcher] Step = 10000 (nb_iteration=52)
...
Progress] Finished 99 / 100 iterations (0.0s)
[MazePerformanceWatcher] Step = 16 (nb_iteration=99)
[Progress] Finished 100 / 100 iterations (0.0s)
[MazePerformanceWatcher] Step = 16 (nb_iteration=100)
[Progress] Completed GPI iteration for 100 times. (total time: 7s)
```

Great!! Agent found the shortest path of blocking maze before and after transofmation.  


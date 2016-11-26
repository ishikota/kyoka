# TickTackToe example
In this example, the goal of agent is to learn the strategy not to lose the game.  
(Because it's known that the best play from both of players leads to draw result)

```
O O X
- X O  <= The player X wins!!
X - -
```

This is simple application but would be good start point for more practical problems.  

In next section, we show how to define *tick-tack-toe* problem as Reinforcement Learning task.

## TickTackToe problem => Reinforcement Learning task
We need to implement following 5 methods to make some problem in RL format.  
### 1. `generate_inital_state()`
```
>>> task.generate_initial_state()
- - -
- - -
- - -
```
### 2.`is_terminal_state(state)`
```
False     True    True
- - -    O O X   X O X
- - -    - X O   X O O
- - -    X - -   O X O
```
### 3. `transit_state(state, action)`
```
state     action     return
- - -                - - -
- - -  +  center =>  - O -
- - -                - - -
```
### 4. `generate_possible_actions(state)`
```
state          returns
- O X
O - X  => [center, upper-left]
O X O
```
### 5.`calculate_reward(state)`
```
if we are player O

 r=0      r=1    r=-1
- - -    O X X   O O X
- - -    X O O   - X O
- - -    X O O   X - -
```

You can checkout complete task implementation [here(task.py)](./task.py).

## Sample code
We prepared sample code to learn strategy by RL algorithms.  
You can checkout sample code under [script/play_with_agent](./script/play_with_agent) directory.  

If you want to try `QLearning` to train agent, run `python examples/ticktacktoe/script/play_with_agent/q_learning.py`.  
After training is finished, you will be asked to play tick-tack-toe with trained agent. type `y`!!  (agent is first player)

```
>>> python examples/ticktacktoe/script/play_with_agent/q_learning.py
[Progress] Start GPI iteration for 1000 times
[EpsilonGreedyAnnealing] Anneal epsilon from 1.0 to 0.1.
[Progress] Finished 1 / 1000 iterations (0.0s)
(some logs are output...)
[Progress] Completed GPI iteration for 1000 times. (total time: 1s)
Do you want to play with trained agent? (y/n) >> y
- - -
- - -
- - O
2: lower_center, 4: lower_left, 8: middle_right, 16: middle_center, 32: middle_left, 64: upper_right, 128: upper_center, 256: upper_left >> 2
- - -
- - -
- X O
```

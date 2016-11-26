# Maze Task example
Simple testbet for reinforcement learning algorithms.  
The goal of agent is to escape from the maze in minimum steps.  

## Sample mazes
We prepared 3 kinds of mazes which are used in the book [Reinforcement Learning: An Introduction](https://webdocs.cs.ualberta.ca/~sutton/book/the-book-2nd.html)  
###1. Dyna Maze
The most simple test bet.  
`S` indicates the start point of maze, `G` is the goal and `X` is the block(agent cannot move on block cell).
```
-------XG
--X----X-
S-X----X-
--X------
-----X---
---------
```

###2. Blocking Maze
After some steps the structure of maze transforms. In transforming, past best path is blocked.  
So agent should realize that maze was transformed and learn another path.

```
 Before         After
--------G     --------G
---------     ---------
---------  => ---------
XXXXXXXX-     -XXXXXXXX
---------     ---------
---S-----     ---S-----
```

###3. Shortcut Maze
After some steps the structure of maze transforms. In transforming, new best path(shortcut) appears.
So agent should realize that the better path is appeared by exploring. 
```
 Before         After
--------G     --------G
---------     ---------
---------  => ---------
-XXXXXXXX     -XXXXXXX-
---------     ---------
---S-----     ---S-----
```

## Sample code
We prepared sample code to try these mazes by RL algorithms.  
You can checkout sample code under [script](./script) directory.  

If you want to try `Dyna Maze` by `montecarlo` method, run `python sample/maze/script/dyna_maze/montecarlo.py`  
After training is finished, the policy which agent learned would be visualized on console like below.
```
>>> python sample/maze/script/dyna_maze/montecarlo.py
[Progress] Start GPI iteration for 100 times
(some logs are output...)
[Progress] Completed GPI iteration for 100 times. (total time: 0s)
[MazePerformanceWatcher] Policy which agent learned is like this.
v>>>vvv-G
v^-^>vv-^
>v->>>v-^
vv->^^>>^
>v<^<-^>^
^>>>>>^>^

```


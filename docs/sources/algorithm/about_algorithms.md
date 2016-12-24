# About Reinforcement Learning Algorithms
All reinforcement learning algorithms implemented in this library have these methods in common


#### `algorithm.setup(task, policy value_function)`
**You must need to call this method before start training.**  

This method sets passed `task`, `policy`, `value_function` on the algorithm.  
So you can access these items by `algorithm.task`, `algorithm.policy`, `algorithm.value_function`.  
The `setup` method of value function is also called in this method.  

---

#### `algorithm.run_gpi(nb_iteration, callbacks=None, verbose=1)`
This method starts training of value function with items passed in `setup` method.  

- `nb_iteration` is the number of episodes of your task played in the training.  
- You can also set `callbacks` objects to interact with task or value function during training.  
- If you want to suppress log of training progress set `verbose=0`

---

#### `algorithm.save(save_dir_path)`, `algorithm.load(load_dir_path)`
You can save and load training results with these methods.

```python
SAVE_DIR_PATH = "~/dev/rl/my_training_result"

algorithm1 = # setup fresh algorithm
# setup training items...
algorithm1.run_gpi(10000)
algorithm1.save(SAVE_DIR_PATH)

algorithm2 = # setup fresh algorithm again
# setup training items again...
algorithm2.load(SAVE_DIR_PATH)
trained_value_function = algorithm2.value_function
```

---

The training code would be like this.

```python
SAVE_DIR_PATH = "~/dev/rl/my_training_result"

task = # Instantiate your learning task object
policy = EpsilonGreedyPolicy(eps=0.9)
policy.set_eps_annealing(initial_eps=0.9, final_eps=0.1, anneal_duration=10000)

algorithm = # Instantiate some algorithm ex. QLearning()
value_function = # Instantiate your value function

callbacks = []
# setup callback objects...

# Do not forget this method before calling "run_gpi"
algorithm.setup(task, policy value_function)

# Load last training result if exists
if os.path.exists(SAVE_DIR_PATH):
    algorithm.load(SAVE_DIR_PATH)

# Start training of value function for 100000 episode
algorithm.run_gpi(100000, callbacks)

# Save training results
algorithm.save(SAVE_DIR_PATH)

```

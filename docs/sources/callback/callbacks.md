# Callbacks implemented by *kyoka*
*kyoka* prepared callbacks which would be useful for reinforcement learning.  

---

## LearningRecorder
Save algorithm in the middle of training in each specified interval.

```python
LearningRecorder(algorithm, root_save_dir_path, save_interval)
```

If you set `root_save_dir_path="dev/rl/training_results`, `save_interval=1000`,  
after 2500 iteration of training, the directory of `root_save_dir_path` has two items like below

```bash
>>> ls dev/rl/training_results
after_1000_iteration        after_2000_iteration
```

If you want to load training results of ater 1000 iteration, you would ...

```python
algorithm.load("dev/rl/training_results/after_1000_iteration")
```

---

## BasePerformanceWatcher
Execute some calculation with task and value function in the middle of training and logs its result.

This class has 2 abstracted methods you need to implement.

- `run_performance_test(self, task, value_function)`: run some calculation and returns its result
- `define_performance_test_interval` : define the interval of training to execute `run_performance_test`

Below implementation checks how much rewards gained in the episode by intermediate value function 
and logs it in each 5000 training iteration.  

```python
from kyoka.callback import BasePerformanceWatcher
from kyoka.algorithm.rl_algorithm import generate_episode
from kyoka.policy import GreedyPolicy

class RewardsPeformanceWatcher(BasePerformanceWatcher):

    def setUp(self, task, value_function):
        self.policy = GreedyPolicy()

    def tearDown(self, task, value_function):
        pass

    def define_performance_test_interval(self):
        return 5000

    def run_performance_test(self, task, value_function):
        episode = generate_episode(task, self.policy, value_function)
        gains = sum([reward for _state, _action, _next_state, reward in episode])
        return gains

    # This is the default implementation to generate log message.
    # So if this implementation is ok, you do not need to implement this method.
    # Argument "test_result" is the item which you returned in "run_performance_test"
    def define_log_message(self, iteration_count, task, value_function, test_result):
        base_msg = "Performance test result : %s (nb_iteration=%d)"
        return base_msg % (test_result, iteration_count)
```

---

## ManualInterruption
You can stop training whenever you want by writing "stop" on specified file.

```python
ManualInterruption(monitor_file_path, watch_interval=30)
```

If you pass `monitor_file_path=dev/rl/stop.txt` then this callback checks

1. if a file exists on `monitor_file_path`
2. if a file exists, find words "stop" in the file
3. if found the word "stop", finish the training

in each 30 iteration of training.  

So you can interrupt training like this.

```bash
echo stop > dev/rl/stop.txt
```

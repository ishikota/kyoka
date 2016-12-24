# What is callback
You can define some procedure which you want to execute in the middle of training through callback object.  
(ex. record performance of value function in each 1000 iteration of training.)  

How to set callback for training is createing list of callbacks and pass it to `algorithm.run_gpi(nb_iteration, callbacks=None)`.  
(You can also pass single callback object (not list) as callbacks arguments)

## Callback methods
The base class of all callback objects is `kyoka.callback.BaseCallback`.  
All callback object must inherit this class and override callback methods as you want.  

`kyoka.callback.BaseCallback` class has 5 callback methods which callback object can override.  

- `before_gpi_start(self, task, value_function)`
    - called when `algorithm.run_gpi` is called
- `before_update(self, iteration_count, task, value_function)`
    - called before `iteration_count` of episode is played in training
- `after_update(self, iteration_count, task, value_function)`
    - called after `iteration_count` of episode is played in training
- `after_gpi_finish(self, task, value_function)`
    - called when training finishes
- `interrupt_gpi(self, iteration_count, task, value_function)`
    - if you return `True` training finishes even if it doesn't reach maximum iteration count

In default, above 4 methods are implemented as empty method and `interrupt_gpi` just returns `False`.  
So callback object don't need to override all methods if nothing to do.

## Logging methods
`kyoka.callback.BaseCallback` class also have utility method `log(message)`.  
If you want to log something on console, we recommend you to use this method instead of `print(message)`.

`log(message)` prints passed message with tag like below.  

```python
>>> callback = WatchIterationCount(5000)
>>> callback.log("Start GPI iteration for 5000 times")
[WatchIterationCount] Start GPI iteration for 5000 times
```

We use class name of callback as tag in default. But you can easily customize it by overriding `define_log_tag` method.  

```
class WatchIterationCount(BaseCallback):
    # some codes...

    def define_log_tag(self):
        return "Progress"

>>> callback = WatchIterationCount(5000)
>>> callback.log("Start GPI iteration for 5000 times")
[Progress] Start GPI iteration for 5000 times
```

---

Here is the sample custom callback to record value of initial state in every 1000 iteration.

```python
import csv
from kyoka.policy import choose_best_action

class InitialStateValueRecorder(BaseCallback):

    def __init__(self, score_file_path):
        self.score_file_path = score_file_path
        self.score_holder = []

    def before_gpi_start(self, task, value_function):
        value = self._predict_value_of_initial_state(task, value_function)
        self.log("Value of initial state is [ %s ]" % value)
        self.score_holder.append(value)

    def after_update(self, iteration_count, task, value_function):
        value = self._predict_value_of_initial_state(task, value_function)
        self.log("Value of initial state is [ %s ]" % value)
        self.score_holder.append(value)

    def after_gpi_finish(self, task, value_function):
        with open(self.score_file_path, "wb") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(self.score_holder)
        self.log("Score is saved on [ %s ]" % self.score_file_path)


    def _predict_value_of_initial_state(self, task, value_function):
        state = task.generate_initial_state()
        action = choose_best_action(task, value_function, state)
        return value_function.predict_value(state, action)
```

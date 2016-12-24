# QLearning - off-policy TD learning method
Qlearning method updates value of state-action pair `Q(s,a)`in following way.
```
s  : current state
a  : action to take at state s choosed by policy PI
r  : reward by transition (s, a)
s' : next state after took action a at s
ga : greedy action at s' under current value function

Q(s,a) = Q(s,a) + alpha [ r + gamma * Q(s', ga) - Q(s, a) ]
```
The new keyword **greedy action** represents the **action which has maximum estimated value** under current value function.
(If multiple actions are greedy then choose one at random.)  
You can get greedy action like this
```python
acts = task.generate_possible_actions(state)
vals = [value_function.predict_value(state, action) for action in acts]
greedy_value_and_actions = [(v,a) for v,a in zip(vals, acts) if v==max(vals)]
_, greedy_action = random.choice(greedy_value_and_actions)
```


This method is also called as *off-policy TD learning* method.  
**off-policy** means that this algorithm use different policy to choose `a` and `a'`.  
QLearning must use *greedy policy* (the policy always choose greedy action) to choose action `a'`.  
But for choosing `a`, you can use any policy. (Most of the case this is *epsilon greedy policy*)

## Algorithm
```
Parameter:
    a  <- alpha. learning rate. [0,1].
    g  <- gamma. discounting factor. [0,1].
Initialize:
    T  <- your RL task
    PI <- policy used in the algorithm
    Q  <- action value function

    Repeat until computational budget runs out:
        S <- generate initial state of task T
        A <- choose action at S by following policy PI
        Repeat until S is terminal state:
            S' <- next state of S after taking action A
            R <- reward gained by taking action A at state S
            A' <- next action at S' by following policy PI
            GA <- greedy action at S' under action value function Q
            Q(S, A) <- Q(S, A) + a * [ R + g * Q(S', GA) - Q(S, A)]
            S, A <- S', A'
```

## Value function
QLearning method provides **tabular** and **approximation** type of value functions.

### QLearningTabularActionValueFunction
If your task is *tabular size*, you can use `QLearningTabularActionValueFunction`.
>If you can store the value of all state-action pair on the memory(array), your task is **tabular** size.

`QLearningTabularActionValueFunction` has 3 abstracted method to define the table size of your task.

- `generate_initial_table` : initialize table object and return it here
- `fetch_value_from_table` : define how to fetch value from your table
- `insert_value_into_table` : define how to insert new value into your table

If the shape of your state-action space is SxA, implementation would be like this.
```python
class MyTabularActionValueFunction(QLearningTabularActionValueFunction):

    def generate_initial_table(self):
        return [[0 for j in range(A)] for i in range(S)]

    def fetch_value_from_table(self, table, state, action):
        return table[state][action]

    def insert_value_into_table(self, table, state, action, new_value):
        table[state][action] = new_value
```

### QLearningApproxActionValueFunction
If your task is not *tabular* size, you use `QLearningApproxActionValueFunction`.

`QLearningApproxActionValueFunction` has 3 abstracted methods. You would wrap some prediction model (ex. neuralnet) in these methods.

- `construct_features` : transform state-action pair into feature representation
- `approx_predict_value` : predict value of state-action pair with prediction model you want to use
- `approx_backup` : update your model in supervised learning way with passed input and output pair

The implementation with some neuralnet library would be like this.
```python
class MyApproxActionValueFunction(QLearningApproxActionValueFunction):

    def setup(self):
        super(MazeApproxActionValueFunction, self).setup()
        self.neuralnet = build_neuralnet_in_some_way()

    def construct_features(self, state, action):
        feature1 = do_something(state, action)
        feature2 = do_anotherthing(state, action)
        return [feature1, feature2]

    def approx_predict_value(self, features):
        return self.neuralnet.predict(features)

    def approx_backup(self, features, backup_target, alpha):
        self.neuralnet.incremental_training(X=features, Y=backup_target)
```

#### Sample code to start learning
```python
test_length = 1000
task = MyTask()
policy = EpsilonGreedyPolicy(eps=0.1)
value_func = MyTabularActionValueFunction()
algorithm = QLearning(gamma=0.99)
algorithm.setup(task, policy, value_func)
algorithm.run_gpi(test_length)
```

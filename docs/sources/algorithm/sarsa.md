# Sarsa - on-policy TD learning method
Sarsa method updates value of state-action pair `Q(s,a)`in following way.
```
s  : current state
a  : action to take at state s choosed by policy PI
r  : reward by transition (s, a)
s' : next state after took action a at s
a' : action to take at state s' choosed by policy PI

Q(s,a) = Q(s,a) + alpha [ r + gamma * Q(s', a') - Q(s, a) ]
```
Update is done with variable `s,a,r,s',a'`. So this algorithm is named as Sarsa.

This method is also called as *on-policy TD learning* method.  The keyword is **on-policy**.  
**on-policy** means that Sarsa uses same policy PI to calculate `a` and `a'`.  
The algorithm uses different policy to select `a` and `a'` is called *off-policy* method. (ex. QLearning)

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
            Q(S, A) <- Q(S, A) + a * [ R + g * Q(S', A') - Q(S, A)]
            S, A <- S', A'
```

## Value function
Sarsa method provides **tabular** and **approximation** type of value functions.

### SarsaTabularActionValueFunction
If your task is *tabular size*, you can use `SarsaTabularActionValueFunction`.
>If you can store the value of all state-action pair on the memory(array), your task is **tabular** size.

`SarsaTabularActionValueFunction` has 3 abstracted method to define the table size of your task.

- `generate_initial_table` : initialize table object and return it here
- `fetch_value_from_table` : define how to fetch value from your table
- `insert_value_into_table` : define how to insert new value into your table

If the shape of your state-action space is SxA, implementation would be like this.
```python
class MyTabularActionValueFunction(SarsaTabularActionValueFunction):

    def generate_initial_table(self):
        return [[0 for j in range(A)] for i in range(S)]

    def fetch_value_from_table(self, table, state, action):
        return table[state][action]

    def insert_value_into_table(self, table, state, action, new_value):
        table[state][action] = new_value
```

### SarsaApproxActionValueFunction
If your task is not *tabular* size, you use `SarsaApproxActionValueFunction`.

`SarsaApproxActionValueFunction` has 3 abstracted methods. You would wrap some prediction model (ex. neuralnet) in these methods.

- `construct_features` : transform state-action pair into feature representation
- `approx_predict_value` : predict value of state-action pair with prediction model you want to use
- `approx_backup` : update your model in supervised learning way with passed input and output pair

The implementation with some neuralnet library would be like this.
```python
class MyApproxActionValueFunction(SarsaApproxActionValueFunction):

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
algorithm = Sarsa(gamma=0.99)
algorithm.setup(task, policy, value_func)
algorithm.run_gpi(test_length)
```

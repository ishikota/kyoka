# MonteCarlo Method
MonteCarlo method estimates the value of action by averaging result of random simulation.  

## Algorithm
Our implementation of MonteCarlo method is the one called as *every-visit MonteCarlo method*.  
**every-visit** means *using every state for update in an episode even if same state appeared in the episode*.

```bash
Parameter:
    g  <- gamma. discounting factor for reward. [0,1]
Initialize:
    T  <- your RL task
    PI <- Policy used to generate episode
    Q  <- action value function

Repeat until computational budget runs out:
    generate an episode of T by following policy PI
    for each state-action pair (S, A)  appeared in the episode:
        G <- sum of rewards gained after state S (G is discounted if g < 1)
        Q(S, A) <- average G of S sampled ever
```

## Reward discounting 
We support *reward discounting* feature.  
If you want to use this feature, set parameter `gamma` in range [0, 1).  
(`gamma` is set to 1 as default value. This means no discounting.)

```python
montecarlo = MonteCarlo(gamma=0.99)
```

Ok, let's see how *reward discounting* works.

Here we assume MonteCarlo method gets the episode of some task like below
```
step0. agent at initial state "s0"
step1. agent take action "a0" at "s0" and move to next state "s1" and received reward 0.
step2. agent take action "a1" at "s1" and move to next state "s2" and received reward 0.
step3. agent take action "a2" at "s2" and move to terminal state "s3" and received reward 1.
```

The value of G at s0 (sum of rewards gained after s0) **without reward discounting** is
```python
G = reward_at_step1 + reward_at_step2 + reward_at_step3
  = 0 + 0 + 1
  = 1
```

The value of G at s0 **with reward discounting** (if `gamma=0.99`) is
```python
G = reward_at_step1 + gamma * reward_at_step2 + gamma**2 * reward_at_step3
  = 0 + 0.1 * 0 + 0.1**2 * 1
  = 0.9801
```

## Value function
MonteCarlo method provides **tabular** and **approximation** type of value functions.  

### MonteCarloTabularActionValueFunction
If your task is *tabular size*, you can use `MonteCarloTabularActionValueFunction`.
>If you can store the value of all state-action pair on the memory(array), your task is **tabular** size.

`MonteCarloTabularActionValueFunction` has 3 abstracted method to define the table size of your task.

- `generate_initial_table` : initialize table object and return it here
- `fetch_value_from_table` : define how to fetch value from your table
- `insert_value_into_table` : define how to insert new value into your table

If the shape of your state-action space is SxA, implementation would be like this.
```python
class MyTabularActionValueFunction(MonteCarloTabularActionValueFunction):

    def generate_initial_table(self):
        return [[0 for j in range(A)] for i in range(S)]

    def fetch_value_from_table(self, table, state, action):
        return table[state][action]

    def insert_value_into_table(self, table, state, action, new_value):
        table[state][action] = new_value
```

### MonteCarloApproxActionValueFunction
If your task is not *tabular* size, you use `MonteCarloApproxActionValueFunction`.

`MonteCarloApproxActionValueFunction` has 3 abstracted methods. You would wrap some prediction model (ex. neuralnet) in these methods.

- `construct_features` : transform state-action pair into feature representation
- `approx_predict_value` : predict value of state-action pair with prediction model you want to use
- `approx_backup` : update your model in supervised learning way with passed input and output pair

The implementation with some neuralnet library would be like this.
```python
class MyApproxActionValueFunction(MonteCarloApproxActionValueFunction):

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
algorithm = MonteCarlo(gamma=0.99)
algorithm.setup(task, policy, value_func)
algorithm.run_gpi(test_length)
```

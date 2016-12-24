# deep Q-learning with experience replay.
Variant of Q-learning for function approximation proposed in the paper  
[Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/abs/nature14236.html?lang=en).

In reinforcement leaning, it's known that function approximation with non-linear model (ex. neuralnet) would be unstable and lead poor learning result.  
To address this problem, *deep Q-learning* combined two key ideas with QLearning.

- Use *experience replay* to reduce correlations between sequence of learning data.
- *Separate taget and behavior network* to stable the source of learning data.

## Algorithm
```
    Parameter
        g <- gamma. discounting factor of QLearning
        N <- capacity of replay memory
        C <- interval to sync Q'(target network) with Q
        minibatch_size <- size of minibatch used to train Q
        replay_start_size <- initial size of replay memory. Fill D
            with this number of experiences which created by random policy.
            This procedure is done in setup phase.

    Initialize
        T  <- your RL task
        PI <- policy used to select action during the episode
        Q  <- approximate action value function (ex. neural network)
        Q' <- target network. initialized by deepcopy Q.
        D  <- filled with replay_start_size of  experiences created by random simulation.
              (experience = tuple of state, action, reward and next_state)

    Repeat until computational budget runs out:
        S <- generate initial state of task T
        A <- choose action at S by following PI
        Repeat until S is terminal state:
            S' <- next state of S after taking action A
            R <- reward gained by taking action A at state S
            A' <- next action at S' by following policy PI
            append experience (S,A,R,S') to D
            MB <- sample minibatch_size of experiences from D
            BT <- transform minibatch of experiences into backup targets
            (BT = [r + g * Q(s', GA) for s,a,r,s' in MB], GA=greedy action at s')
            Update Q by using BT (minibatch of backup targets)
            Every C step: Q' <- Q (ex. deepcopy weights of Q to Q')
            S, A <- S', A'
```

## Value function
`DeepQLearning` method provides only **approximation** type of value functions.

### DeepQLearningApproxActionValueFunction
`DeepQLearningApproxActionValueFunction` has 6 abstracted methods. You would wrap your prediction model (ex. neuralnet) in these methods.

- `initialize_network` : initialize your prediction model here
- `deepcopy_network` : define how to create deepcopy of your prediction model
- `predict_value_by_network` : predict value of state-action pair by your prediction model
- `backup_on_minibatch` : train your prediction model with passed learning minibatch
- `save_networks` : save your prediction model as you like (ex. save the weights of neuralnet)
- `load_networks` : load your prediction model from resource created by `save_networks`

The implementation with some neuralnet library would be like this.
```python
class MyApproxActionValueFunction(DeepQLearningApproxActionValueFunction):

    # the model returned here is used as "q_network" (Q of above algorithm)
    def initialize_network(self):
        model = build_neuralnet()
        return model

    # the model returned here is used as "q_hat_network" (Q' of above algorithm)
    def deepcopy_network(self, q_network):
        original_weight = q_network.get_weights()
        deepcopy_network = self.initialize_network()
        deepcopy_network.set_weights(original_weight)
        return deepcopy_network

    # return prediction value of passed state action pair.
    # passed network would be "q_network" or "q_hat_network".
    def predict_value_by_network(self, network, state, action):
        features = build_features(state, action)
        prediction = network.predict(features)
        return prediction

    # train passed q_network with backup_minibatch
    # you would need to transform backup_minibatch into input output pair like
    # supervised learning format.
    def backup_on_minibatch(self, q_network, backup_minibatch):
        # backup_minibatch is array of (state, action, target_value).
        X = [build_features(state, action) for state, action, _target in backup_minibatch]
        y = [target for _state, _action, target in backup_minibatch]
        q_network.train_on_minibatch(X, y)

    # save passed two neuralnet on passed directory
    def save_networks(self, q_network, q_hat_network, save_dir_path):
        q_network.save_weights("%s/q_weight.h5" % save_dir_path)
        q_hat_network.save_weights("%s/q_hat_weight.h5" % save_dir_path)

    # load "q_network" and "q_hat_network" from passed directory and return them in
    # "q_network", "q_hat_network" order.
    def load_networks(self, load_dir_path):
        q_network = self.initialize_network()
        q_network.load_weights("%s/q_weight.h5" % load_dir_path)
        q_hat_network = self.initialize_network()
        q_hat_network.load_weights("%s/q_hat_weight.h5" % load_dir_path)
        return q_network, q_hat_network

```

#### Sample code to start learning
```python
TEST_LENGTH = 5000000
task = MyTask()
policy = EpsilonGreedyPolicy(eps=1.0)
policy.set_eps_annealing(initial_eps=1.0, final_eps=0.1, anneal_duration=1000000)
value_func = MyApproxActionValueFunction()
algorithm = DeepQLearning(gamma=0.99, N=100000, C=1000, minibatch_size=32, replay_start_size=50000)
algorithm.setup(task, policy, value_func)
algorithm.run_gpi(test_length)
```


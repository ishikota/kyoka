import random

from kyoka.utils import build_not_implemented_msg

def choose_best_action(task, value_function, state, rand=None):
    rand = rand if rand else random
    actions = task.generate_possible_actions(state)
    Q_value_for_actions = [value_function.predict_value(state, action) for action in actions]
    max_Q_value = max(Q_value_for_actions)
    Q_act_pair = zip(Q_value_for_actions, actions)
    best_actions = [act for Q_value, act in Q_act_pair if max_Q_value == Q_value]
    best_action = rand.choice(best_actions)
    return best_action

class BasePolicy(object):

    def choose_action(self, task, value_function, state, action):
        err_msg = build_not_implemented_msg(self, "choose_action")
        raise NotImplementedError(err_msg)

class GreedyPolicy(BasePolicy):

    def __init__(self, rand=None):
        self.rand = rand if rand else random

    def choose_action(self, task, value_function, state):
        return choose_best_action(task, value_function, state, self.rand)

class EpsilonGreedyPolicy(BasePolicy):

    def __init__(self, eps=0.05, rand=None):
        self.eps = eps
        self.rand = rand if rand else random
        self.do_annealing = False

    def choose_action(self, task, value_function, state):
        actions = task.generate_possible_actions(state)
        best_action = choose_best_action(task, value_function, state, self.rand)
        probs = self.__calc_select_probability(best_action, actions)
        selected_action_idx = self.__roulette(probs)
        return actions[selected_action_idx]

    def set_eps_annealing(self, initial_eps, final_eps, anneal_duration):
        self.do_annealing = True
        self.eps = initial_eps
        self.min_eps = final_eps
        self.anneal_step = (initial_eps - final_eps) / anneal_duration

    def anneal_eps(self):
        self.eps = max(self.min_eps, self.eps - self.anneal_step)

    def __calc_select_probability(self, best_action, actions):
        e = self.eps / len(actions)
        bonus = 1 - self.eps
        calc_prob = lambda action: e + bonus if action == best_action else e
        return [calc_prob(action) for action in actions]

    def __roulette(self, probs):
        hit_idx = -1
        prob_sum = 0
        dart = self.rand.random()
        for idx, prob in enumerate(probs):
            prob_sum += prob
            if dart < prob_sum:
                hit_idx = idx
                break
        return hit_idx


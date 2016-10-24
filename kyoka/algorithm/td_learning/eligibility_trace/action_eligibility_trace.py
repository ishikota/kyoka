from collections import defaultdict

class ActionEligibilityTrace:

  TYPE_ACCUMULATING = "accumulating_trace"
  TYPE_REPLACING = "replacing_trace"

  def __init__(self, update_type, discard_threshold=0.0001, gamma=0.99, lambda_=0.9):
    self.__validate_update_type(update_type)
    self.update_type = update_type
    self.discard_threshold = discard_threshold
    self.gamma = gamma
    self.lambda_ = lambda_
    self.eligibility_holder = self.__generate_action_eligibility_holder()

  def get(self, state, action):
    return self.eligibility_holder[state][action]

  def update(self, state, action):
    if self.TYPE_ACCUMULATING == self.update_type:
      self.__update(state, action, self.get(state, action) + 1)
    elif self.TYPE_REPLACING == self.update_type:
      self.__update(state, action, 1)
    else:
      self.__validate_update_type(self.update_type)

  def decay(self, state, action):
    decayed = self.gamma * self.lambda_ * self.get(state, action)
    self.__update(state, action, decayed)
    if self.get(state, action) <= self.discard_threshold:
      self.__discard(state, action)

  def get_eligibilities(self):
    eligibilities = []
    for state in self.eligibility_holder:
      for action in self.eligibility_holder[state]:
        eligibility = self.eligibility_holder[state][action]
        eligibilities.append((state, action, eligibility))
    return eligibilities

  def clear(self):
    self.eligibility_holder = self.__generate_action_eligibility_holder()

  def dump(self):
    return (self.update_type, self.discard_threshold,\
            self.gamma, self.lambda_, self.get_eligibilities())

  def load(self, serial):
    self.clear()
    self.update_type, self.discard_threshold, self.gamma, self.lambda_, eligibilities = serial
    for state, action, eligibility in eligibilities:
      self.__update(state, action, eligibility)

  def __validate_update_type(self, update_type):
    if not update_type in [self.TYPE_ACCUMULATING, self.TYPE_REPLACING]:
      raise TypeError("unknown update type [ %s ] passed" % update_type)

  def __update(self, state, action, new_value):
    self.eligibility_holder[state][action] = new_value

  def __discard(self, state, action):
    return self.eligibility_holder[state].pop(action)

  def __generate_action_eligibility_holder(self):
    return defaultdict(lambda: defaultdict(float))


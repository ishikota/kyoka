import random

class ExperienceReplay(object):

  def __init__(self, max_size):
    self.max_size = max_size
    self.queue = []

  def store_transition(self, state, action, reward, next_state):
    if len(self.queue) >= self.max_size:
      self.queue.pop(0)
    self.queue.append((state, action, reward, next_state))

  def sample_minibatch(self, minibatch_size):
    return random.sample(self.queue, minibatch_size)

  def dump(self):
    return (self.max_size, self.queue)

  def load(self, serial):
    self.max_size, self.queue = serial


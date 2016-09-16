class BasePolicy(object):

  def __init__(self, domain):
    self.domain = domain

  def choose_action(self, Q, state):
    raise NotImplementedError("[%s] class does not implement [choose_action] method" % self.__class__.__name__)


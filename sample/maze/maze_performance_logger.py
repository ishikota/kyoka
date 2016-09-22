from kyoka.callback.base_callback import BaseCallback
from sample.maze.maze_helper import MazeHelper

class MazePerformanceLogger(BaseCallback):

  def before_gpi_start(self, domain, value_function):
    self.step_log = []

  def after_update(self, iteration_count, domain, value_function, delta):
    step_to_goal = MazeHelper.measure_performance(domain, value_function)
    self.step_log.append(step_to_goal)


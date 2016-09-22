from kyoka.callback.base_callback import BaseCallback
import logging

class MazeTransformer(BaseCallback):

  def __init__(self):
    BaseCallback.__init__(self)
    self.transformation = {}

  def set_transformation(self, timing, maze_path):
    self.transformation[timing] = maze_path

  def before_update(self, iteration_count, domain, value_function):
    if iteration_count in self.transformation:
      maze_filepath = self.transformation[iteration_count]
      domain.read_maze(maze_filepath)
      logging.debug("Maze transformed into [ %s ]" % maze_filepath)


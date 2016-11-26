import logging

from kyoka.callback import BaseCallback, BasePerformanceWatcher
from examples.maze.helper import measure_performance, visualize_policy

class MazePerformanceWatcher(BasePerformanceWatcher):

    def define_performance_test_interval(self):
        return 1

    def run_performance_test(self, task, value_function):
        step_to_goal = measure_performance(task, value_function)
        policy = visualize_policy(task, value_function)
        return step_to_goal, policy

    def define_log_message(self, iteration_count, task, value_function, test_result):
        step_to_goal, _ = test_result
        return "Step = %d (nb_iteration=%d)" % (step_to_goal,iteration_count)

    def tearDown(self, task, value_function):
        msg_prefix = "Policy which agent learned is like this.\n"
        self.log(msg_prefix + visualize_policy(task, value_function))

class MazeTransformer(BaseCallback):

    def __init__(self):
        BaseCallback.__init__(self)
        self.transformation = {}

    def set_transformation(self, timing, maze_path):
        self.transformation[timing] = maze_path

    def before_update(self, iteration_count, task, value_function):
        if iteration_count in self.transformation:
            maze_filepath = self.transformation[iteration_count]
            task.read_maze(maze_filepath)
            logging.debug("Maze transformed into [ %s ]" % maze_filepath)


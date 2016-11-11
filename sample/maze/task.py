from kyoka.task import BaseTask

class MazeTask(BaseTask):

    UP    = 0
    DOWN  = 1
    RIGHT = 2
    LEFT  = 3

    def read_maze(self, filepath):
        with open(filepath, 'rb') as f:
            self.maze = [line.rstrip() for line in f.readlines()]
            self.validate_maze(self.maze)

    def get_maze_shape(self):
        width, height = len(self.maze[0]), len(self.maze)
        return (height, width)

    def generate_initial_state(self):
        return self.__find_cell('S')

    def is_terminal_state(self, state):
        row, col = state
        return 'G' == self.maze[row][col]

    def transit_state(self, state, action):
        row, col = state
        height, width = self.get_maze_shape()
        if action == self.UP:
            row = max(0, row-1)
        elif action == self.DOWN:
            row = min(height-1, row+1)
        elif action == self.RIGHT:
            col= min(width-1, col+1)
        elif action == self.LEFT:
            col = max(0, col-1)
        if 'X' != self.maze[row][col]:
            return (row, col)
        else:
            return state

    def generate_possible_actions(self, state):
        return [self.UP, self.DOWN, self.RIGHT, self.LEFT]

    def calculate_reward(self, state):
        row, col = state
        return 1 if 'G' == self.maze[row][col] else 0


    def validate_maze(self, maze):
        cells = reduce(lambda x, y: x+y, maze)
        start_count = cells.count('S')
        goal_count = cells.count('G')
        if start_count != 1 or goal_count != 1:
            raise ValueError("Invalid maze passed. reason: start_count=%d, goal_count=%d")

    def generate_terminal_state(self):
        return self.__find_cell('G')

    def __find_cell(self, target):
        for row in range(len(self.maze)):
            for col in range(len(self.maze[0])):
                if self.maze[row][col] == target: return (row, col)


from tests.base_unittest import BaseUnitTest
from sample.ticktacktoe.ticktacktoe_domain import TickTackToeDomain

class TickTackToeDomainTest(BaseUnitTest):
  """
  Board and bit-flg relation
  EX1.
    (O) first_player_board  = 110000000     OO-
    (X) second_player_board = 000000011  => ---
                                            -XX
  EX2.
    (O) first_player_board  = 101010101     OXO
    (X) second_player_board = 010101010  => XOX
                                            OXO
  """

  def setUp(self):
    self.domain = TickTackToeDomain()

  def test_generate_inital_state(self):
    state = self.domain.generate_initial_state()
    self.eq((0,0), state)

  def test_terminal_state_check_when_initial_state(self):
    state = self.__gen_initial_state()
    self.false(self.domain.is_terminal_state(state))

  def test_terminal_state_check_when_draw(self):
    state = self.__gen_draw_state()
    self.true(self.domain.is_terminal_state(state))

  def test_terminal_state_check_when_vertical_win(self):
    state = self.__gen_vertical_win_state()
    self.true(self.domain.is_terminal_state(state))

  def test_terminal_state_check_when_horizontal_win(self):
    state = self.__gen_horizontal_win_state()
    self.true(self.domain.is_terminal_state(state))

  def test_terminal_state_check_when_diagonal_win(self):
    state = self.__gen_diagonal_win_state()
    self.true(self.domain.is_terminal_state(state))

  def test_transit_state_from_initial_state(self):
    state = self.__gen_initial_state()
    state = self.domain.transit_state(state, 1)
    state = self.domain.transit_state(state, 2)
    state = self.domain.transit_state(state, 256)
    expected = (257, 2)
    self.eq(expected, state)

  def test_generate_possible_actions_when_draw(self):
    state = self.__gen_draw_state()
    expected = []
    self.eq(expected, self.domain.generate_possible_actions(state))

  def test_generate_possible_actions_when_initial_state(self):
    state = self.__gen_initial_state()
    expected = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    self.eq(expected, self.domain.generate_possible_actions(state))

  def test_generate_possible_actions_when_vertical_win(self):
    state = self.__gen_vertical_win_state()
    expected = [1, 2, 8, 64]
    self.eq(expected, self.domain.generate_possible_actions(state))

  def test_generate_possible_actions_when_horizontal_win(self):
    state = self.__gen_horizontal_win_state()
    expected = [1, 2, 4, 8]
    self.eq(expected, self.domain.generate_possible_actions(state))

  def test_generate_possible_actions_when_diagonal_win(self):
    state = self.__gen_diagonal_win_state()
    expected = [2, 4, 8, 32]
    self.eq(expected, self.domain.generate_possible_actions(state))

  def test_calculate_reward_when_initial_state(self):
    state = self.__gen_initial_state()
    self.eq(TickTackToeDomain.REWARD_NONE, self.domain.calculate_reward(state))

  def test_calculate_reward_when_first_player_win(self):
    state = self.__gen_diagonal_win_state()
    self.eq(TickTackToeDomain.REWARD_WIN, self.domain.calculate_reward(state))

  def test_calculate_reward_when_first_player_lose(self):
    state = self.__gen_diagonal_win_state()[::-1]
    self.eq(TickTackToeDomain.REWARD_LOSE, self.domain.calculate_reward(state))

  def test_is_first_player_flg(self):
    domain = TickTackToeDomain(is_first_player=False)
    state = self.__gen_diagonal_win_state()
    self.eq(TickTackToeDomain.REWARD_LOSE, domain.calculate_reward(state))


  def __gen_initial_state(self):
    """
    ---
    ---
    ---
    """
    bin2i = lambda b: int(b, 2)
    first_player_board = bin2i("000000000")
    second_player_board = bin2i("000000000")
    state = (first_player_board, second_player_board)
    return state

  def __gen_draw_state(self):
    """
    OXO
    OXO
    XOX
    """
    bin2i = lambda b: int(b, 2)
    first_player_board = bin2i("010101101")
    second_player_board = bin2i("101010010")
    state = (first_player_board, second_player_board)
    return state

  def __gen_vertical_win_state(self):
    """
    OX-
    OX-
    O--
    """
    bin2i = lambda b: int(b, 2)
    first_player_board = bin2i("100100100")
    second_player_board = bin2i("010010000")
    state = (first_player_board, second_player_board)
    return state

  def __gen_horizontal_win_state(self):
    """
    OOO
    XX-
    ---
    """
    bin2i = lambda b: int(b, 2)
    first_player_board = bin2i("111000000")
    second_player_board = bin2i("000110000")
    state = (first_player_board, second_player_board)
    return state

  def __gen_diagonal_win_state(self):
    """
    OXX
    -O-
    --O
    """
    bin2i = lambda b: int(b, 2)
    first_player_board = bin2i("100010001")
    second_player_board = bin2i("011000000")
    state = (first_player_board, second_player_board)
    return state


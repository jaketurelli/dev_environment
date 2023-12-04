import copy
import random


class tic_tac_toe():
    def __init__(self):
        self.reset()
        self.winning_combos = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6],
            [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
        self.winning_combo_per_move = {
            move: [combo for combo in self.winning_combos if move in combo]
            for move in range(9)}

    def reset(self):
        self.board = [' ' for _ in range(9)]
        self.history = []
        self.winner = None

    def as_str(self, items=None):
        if items is None:
            items = self.history
        if isinstance(items, list) and items and not isinstance(items[0], list):
            items = [items]
        boards_str = '\n'.join(
            [''.join([f'{n}┌───┬───┬───┐' for n in range(1, 1 + len(items))]),
             ''.join([f' │ {b[0]} │ {b[1]} │ {b[2]} │' for b in items]),
             ''.join([' ├───┼───┼───┤' for _ in items]),
             ''.join([f' │ {b[3]} │ {b[4]} │ {b[5]} │' for b in items]),
             ''.join([' ├───┼───┼───┤' for _ in items]),
             ''.join([f' │ {b[6]} │ {b[7]} │ {b[8]} │' for b in items]),
             ''.join([' └───┴───┴───┘' for _ in items])])
        if len(items) > 2:
            if self.winner is None:
                boards_str += ' Draw!'
            else:
                boards_str += f' {self.winner} WON!'
        return boards_str

    def get_available_positions(self):
        return [i for i, x in enumerate(self.board) if x == ' ']

    def check_if_move_won(self, choice, marker):
        for combo in self.winning_combo_per_move[choice]:
            if all(self.board[i] == marker for i in combo):
                return True
        return False

    def random_player(self):
        return random.choice(self.get_available_positions())

    def play(self, player_x=None, player_o=None, start_first=None):
        self.reset()
        if player_x is None:
            player_x = self.random_player
        if player_o is None:
            player_o = self.random_player
        if start_first is None:
            start_first = random.randint(0, 1)
        for i in range(9):
            marker = 'X' if i % 2 == start_first else 'O'
            choice = player_x() if marker == 'X' else player_o()
            self.board[choice] = marker
            self.history += [copy.deepcopy(self.board)]
            if self.check_if_move_won(choice, marker):
                self.winner = marker
                break


if __name__ == "__main__":

    game = tic_tac_toe()
    max_games = 3
    for i in range(max_games):
        print(f"Game {i+1}/{max_games}:")
        game.play()
        print(game.as_str())

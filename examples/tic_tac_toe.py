"""train a Q-learning agent and play a game of tic tac toe against it"""
import copy
import random

_all_winning_combos = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6],
    [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
_winning_combo_per_move = {
    move: [combo for combo in _all_winning_combos if move in combo]
    for move in range(9)}


class _colors():
    RED = "\033[0;31m"
    BLUE = "\033[0;34m"
    FAINT = "\033[2m"
    END = "\033[0m"


def _clr(x):
    if x == 'X':
        return _colors.RED + x + _colors.END
    if x == 'O':
        return _colors.BLUE + x + _colors.END
    return _colors.FAINT + x + _colors.END


class tic_tac_toe():

    def __init__(self):
        self.reset()
        self.max_transformations = 7

    def reset(self):
        self.reset_game()
        self.wins = {'X': 0, 'O': 0}
        self.draws = 0
        self.games = 0

    def reset_game(self):
        self.board = ' ' * 9
        self.history = []
        self.state_history = {'X': [], 'O': []}
        self.action_history = {'X': [], 'O': []}
        self.winner = None

    def stats_str(self):
        return 'Stats:\n' + \
            f'    {self.wins["X"]:>5}/{self.games} ({self.wins["X"]/self.games*100:>5.1f}%) X wins\n' +\
            f'    {self.wins["O"]:>5}/{self.games} ({self.wins["O"]/self.games*100:>5.1f}%) O wins\n' +\
            f'    {self.draws:>5}/{self.games} ({self.draws/self.games*100:>5.1f}%) draws'

    def as_str(self, items=None):
        if items is None:
            items = self.history
        if not isinstance(items, list):
            items = [items]
        boards_str = '\n'.join(
            [''.join([f'{n}┌───┬───┬───┐' for n in range(1, 1 + len(items))]),
             ''.join([f' │ {_clr(b[0])} │ {_clr(b[1])} │ {_clr(b[2])} │' for b in items]),
             ''.join([' ├───┼───┼───┤' for _ in items]),
             ''.join([f' │ {_clr(b[3])} │ {_clr(b[4])} │ {_clr(b[5])} │' for b in items]),
             ''.join([' ├───┼───┼───┤' for _ in items]),
             ''.join([f' │ {_clr(b[6])} │ {_clr(b[7])} │ {_clr(b[8])} │' for b in items]),
             ''.join([' └───┴───┴───┘' for _ in items])])
        if len(items) > 2:
            winner = self.check_for_winner()
            if winner is None:
                if len(items) == 9:
                    boards_str += ' Draw!'
            else:
                boards_str += f' {winner} WON!'
        return boards_str

    def get_available_positions(self):
        return [i for i, x in enumerate(self.board) if x == ' ']

    def check_for_winner(self):
        for marker in 'XO':
            for combo in _all_winning_combos:
                if all(self.board[i] == marker for i in combo):
                    return marker
        return None

    def check_if_move_won(self, choice, marker):
        for combo in _winning_combo_per_move[choice]:
            if all(self.board[i] == marker for i in combo):
                return True
        return False

    def random_player(self, state, available_actions):
        if available_actions:
            return random.choice(available_actions)

    def human_player(self, _, available_actions, role=None):
        hist_with_options = copy.deepcopy(self.history)
        if not hist_with_options:
            hist_with_options += [' ' * 9]
        hist_with_options[-1] = ''.join(
            str(x) if x in available_actions else hist_with_options[-1][x] for x in range(9))
        print(self.as_str(hist_with_options))
        if not available_actions:
            return
        if len(available_actions) == 1:
            return available_actions[0]
        while True:
            try:
                role_str = f'You are playing as {_clr(role)}. ' if role else ''
                choice = input(f'{role_str}Choose integer move or Ctrl+C to exit.')
            except KeyboardInterrupt:
                print('\nExiting.')
                exit(0)
            try:
                choice = int(choice)
                if choice in available_actions:
                    return choice
                print('not an available move')
            except ValueError:
                print('not an integer')

    def ordered_player(self, _, available_actions):
        if not available_actions:
            return None
        return available_actions[0]

    def get_state(self, self_marker):
        opponent_marker = 'X' if self_marker == 'O' else 'O'
        return self.board.replace(self_marker, '1').replace(opponent_marker, '0')

    @staticmethod
    def step_state_and_flip_player(state, action):
        return (state[:action] + '1' + state[action + 1:]).replace('0', '$').replace('1', '0').replace('$', '1')

    def state_transformations(self, state, i):
        if i == 0:
            return ''.join(state[int(x)] for x in '630741852')  # clockwise 90 deg
        if i == 1:
            return ''.join(state[int(x)] for x in '876543210')  # clockwise 180 deg
        if i == 2:
            return ''.join(state[int(x)] for x in '258147036')  # clockwise 270 deg
        if i == 3:
            return ''.join(state[int(x)] for x in '210543876')  # mirror vertical axis
        if i == 4:
            return ''.join(state[int(x)] for x in '678345012')  # mirror horizontal axis
        if i == 5:
            return ''.join(state[int(x)] for x in '036147258')  # mirror diagonal axis starting top left
        if i == 6:
            return ''.join(state[int(x)] for x in '852741630')  # mirror diagonal axis starting top right

    def action_transformations(self, move, i):
        state = [' ' for i in range(9)]
        state[move] = '1'
        if i == 0:
            return ''.join(state[int(x)] for x in '630741852').index('1')  # clockwise 90 deg
        if i == 1:
            return ''.join(state[int(x)] for x in '876543210').index('1')  # clockwise 180 deg
        if i == 2:
            return ''.join(state[int(x)] for x in '258147036').index('1')  # clockwise 270 deg
        if i == 3:
            return ''.join(state[int(x)] for x in '210543876').index('1')  # mirror vertical axis
        if i == 4:
            return ''.join(state[int(x)] for x in '678345012').index('1')  # mirror horizontal axis
        if i == 5:
            return ''.join(state[int(x)] for x in '036147258').index('1')  # mirror diagonal axis starting top left
        if i == 6:
            return ''.join(state[int(x)] for x in '852741630').index('1')  # mirror diagonal axis starting top right

    def play(self, player_x=None, player_o=None, start_first=None, learner=None):
        self.reset_game()
        if player_x is None:
            player_x = self.random_player
        if player_o is None:
            player_o = self.random_player
        if start_first is None:
            start_first = random.randint(0, 1)
        else:
            start_first = 0 if start_first == 'X' else 1
        for i in range(9):
            marker = 'X' if i % 2 == start_first else 'O'
            state = self.get_state(marker)
            available_actions = self.get_available_positions()
            choice = player_x(state, available_actions) if marker == 'X' else player_o(state, available_actions)
            if choice not in available_actions:
                raise RuntimeError('need a valid move')
            self.board = self.board[:choice] + marker + self.board[choice + 1:]
            self.history += [copy.deepcopy(self.board)]
            self.state_history[marker] += [state]
            self.action_history[marker] += [choice]
            if i > 2 and self.check_if_move_won(choice, marker):
                self.winner = marker
                break

        self.games += 1
        if self.winner:
            self.wins[self.winner] += 1
        else:
            self.draws += 1

        # let the player see the end state
        for i in range(9, 11):
            marker = 'X' if i % 2 == start_first else 'O'
            state = self.get_state(marker)
            choice = player_x(state, []) if marker == 'X' else player_o(state, [])

        # pass the information to the learner
        if learner is not None:
            for marker in 'XO':
                reward = 0.
                if self.winner is not None:
                    reward = 1. if self.winner == marker else -1.

                # the reward only applies to the last action
                reward_hist = [0. for _ in self.action_history[marker]]
                reward_hist[-1] = reward
                state_hist, action_hist = self.state_history[marker], self.action_history[marker]

                # learn move
                for state, action, reward in zip(reversed(state_hist), reversed(action_hist), reversed(reward_hist)):
                    learner(state, action, reward)

                # learn equivalent moves
                for i in range(self.max_transformations):
                    state_hist_t = [self.state_transformations(s, i) for s in state_hist]
                    action_hist_t = [self.action_transformations(s, i) for s in action_hist]
                    for state, action, reward in zip(reversed(state_hist_t), reversed(action_hist_t), reversed(reward_hist)):
                        learner(state, action, reward)


class q_learning():
    def __init__(self, opponent_state_predictor):
        self.R = 0.1  # learning rate
        self.invR = 1. - self.R  # inverse learning rate
        self.D = 1.  # discount factor
        self.Q = {}
        # must conform to: opponent_state = opponent_state_predictor(state, action)
        self.opponent_state_predictor = opponent_state_predictor

    def estimate_future_max_q(self, state, action):
        future_reward = 0.
        future_opponent_reward = 0.
        is_opponent = False
        while True:
            state = self.opponent_state_predictor(state, action)
            is_opponent = not is_opponent
            actions = self.Q.get(state)
            if actions is None:
                break
            action = max(actions, key=actions.get)
            best_actions = [c for c, v in actions.items() if v == actions[action]]
            if len(best_actions) == 1:
                action = best_actions[0]
            else:
                action = random.choice(best_actions)
            if is_opponent:
                future_opponent_reward = max(future_opponent_reward, self.Q[state][action])
            else:
                future_reward = max(future_reward, self.Q[state][action])
        return future_reward - future_opponent_reward

    def ensure_Q_is_initialized(self, state, action, init_val):
        if state not in self.Q:
            self.Q[state] = {}
        if action not in self.Q[state]:
            self.Q[state][action] = init_val

    def learn(self, state, action, reward):
        self.ensure_Q_is_initialized(state, action, 0.)
        self.Q[state][action] = self.invR * self.Q[state][action] + \
            self.R * (reward + self.D * self.estimate_future_max_q(state, action))

    def deploy(self, state, available_actions):
        if not available_actions:
            return None
        actions = self.Q.get(state)
        if not actions:
            return random.choice(available_actions)
        untried_choices = [a for a in available_actions if a not in actions]
        if untried_choices:
            return random.choice(untried_choices)
        best_action = max(actions, key=actions.get)
        best_actions = [a for a, r in actions.items() if r == actions[best_action]]
        if len(best_actions) == 1:
            return best_action
        return random.choice(best_actions)

    def state_weights_str(self, state=' ' * 9):
        state_dict = copy.deepcopy(self.Q.get(state, {}))
        for i in range(9):
            if i not in state_dict:
                state_dict[i] = None
        d = [(k, f'  {"+" if state[k] == "1" else "-"}  ' if v is None else f'{v:>5.2f}') for k, v in state_dict.items()]
        d.sort()
        d = [x[1] for x in d]
        return f'''
┌─────┬─────┬─────┐
│{d[0]}│{d[1]}│{d[2]}│
├─────┼─────┼─────┤
│{d[3]}│{d[4]}│{d[5]}│
├─────┼─────┼─────┤
│{d[6]}│{d[7]}│{d[8]}│
└─────┴─────┴─────┘
'''


if __name__ == "__main__":

    game = tic_tac_toe()

    learners = [q_learning(opponent_state_predictor=tic_tac_toe.step_state_and_flip_player) for _ in range(3)]

    print('\nStarting Random and Self-Training')
    training_runs = 100
    for i in range(training_runs):
        for ii in range(len(learners)):

            # observe two random players playing
            game.play(learner=learners[ii].learn)

            # observe self-play
            for iii in range(len(learners)):
                j = (iii + 1) % len(learners)
                game.play(
                    learner=learners[ii].learn,
                    player_x=learners[iii].deploy,
                    player_o=learners[j].deploy)
                game.play(
                    learner=learners[ii].learn,
                    player_x=learners[iii].deploy)

        # print progress
        if (i + 1) % (training_runs // 10) == 0:
            print(f'    {(i+1)/training_runs*100:3.1f}% runs complete')
    print('Training ' + game.stats_str())

    print('\nStarting Competition/Assessment')
    results = []
    for li, L in enumerate(learners):
        game.reset()
        for i in range(1000):
            game.play(learner=L.learn, player_x=L.deploy)
        score = game.wins['X'] - game.wins['O'] - game.draws * 0.1
        results += [(score, li, game.stats_str())]
    results.sort(reverse=True)

    _, winner, winning_stats = results[0]
    print(f"Rankings: {[x[0] for x in results]}")
    print("Winner's " + winning_stats)
    learner = learners[winner]

    print(f'States Explored: {len(learner.Q)}/4520')

    print('\nStarting User Play:')
    if input('play? y/n:\n').lower() == 'y':
        game.reset()
        while True:

            def player(s, a):
                new_move = game.human_player(s, a, role='X')
                if new_move is not None:
                    print("What did your opponent think about before it moved?")
                    print(learner.state_weights_str(learner.opponent_state_predictor(game.get_state('X'), new_move)))
                return new_move

            prior_weights = learner.state_weights_str()
            game.play(
                learner=learner.learn,
                player_x=player,
                player_o=learner.deploy,
                start_first='X')
            new_weights = learner.state_weights_str()
            weight_change = '\n'.join([f'{a}  -->  {b}' for a, b in zip(prior_weights.strip().split('\n'), new_weights.strip().split('\n'), )])
            print(f'Change in first layer weights:\n\n{weight_change}')
            print(game.stats_str())

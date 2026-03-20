import torch
import numpy as np

from .utils import load_words, feedback, filter_candidates, encode_state
from .policy import PolicyNet


class Evaluator:

    def __init__(self, model_path):

        self.guess_words = load_words("./data/words.txt")
        self.answer_words = load_words("./data/test_words.txt")

        self.policy = PolicyNet(157, len(self.guess_words))
        self.policy.load_state_dict(torch.load(model_path))
        self.policy.eval()

    def play_game(self, target):

        candidates = list(self.answer_words)

        for step in range(6):

            state = torch.tensor(
                encode_state(candidates),
                dtype=torch.float32
            )

            with torch.no_grad():
                logits = self.policy(state)

            action = torch.argmax(logits).item()

            guess = self.guess_words[action]

            pattern = feedback(guess, target)

            candidates = filter_candidates(candidates, guess, pattern)

            if guess == target:
                return True, step+1

        return False, 6

    def evaluate(self, n_games=500):

        wins = 0
        guesses = []

        for _ in range(n_games):

            target = np.random.choice(self.answer_words)

            win, steps = self.play_game(target)

            if win:
                wins += 1

            guesses.append(steps)

        print("\n=== RESULTS ===")
        print("Win rate:", wins / n_games)
        print("Avg guesses:", np.mean(guesses))
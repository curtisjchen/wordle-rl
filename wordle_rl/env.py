import numpy as np
from .utils import feedback, filter_candidates, encode_state


class WordleEnv:

    def __init__(self, guess_words, answer_words):

        self.guess_words = guess_words
        self.answer_words = answer_words

    def reset(self):

        self.target = np.random.choice(self.answer_words)

        self.candidates = list(self.answer_words)

        self.steps = 0

        return encode_state(self.candidates)

    def step(self, guess):

        self.steps += 1

        pattern = feedback(guess, self.target)

        before = len(self.candidates)

        self.candidates = filter_candidates(self.candidates, guess, pattern)

        after = len(self.candidates)

        reward = np.log(before+1) - np.log(after+1)

        done = False

        if guess == self.target:
            reward += 5
            done = True

        if self.steps >= 6:
            done = True

        return encode_state(self.candidates), reward, done
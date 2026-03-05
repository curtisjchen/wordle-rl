"""
wordle_env.py — Pure Python Wordle environment (no Gymnasium required).

Observation:  int32 array of shape (61,)
    [0  :30]  letter indices per tile   (0-25 = a-z,  26 = empty)
    [30 :60]  color  indices per tile   (0=gray, 1=yellow, 2=green, 3=empty)
    [60]      current step              (0-6)

    Integer indices (not normalized floats) so the network can use
    learned embeddings for letters and colors.

Reward (per step):
    STEP_PENALTY                          always         (-0.01)
    INFO_GAIN_COEF * log(before/after)    always         (information gain)

Reward (terminal):
    WIN_REWARDS[step_num]                 on win         (32 to 1 exponential)
    LOSS_REWARD                           on loss        (-6.0)
"""

import os
import random
import numpy as np


# Reward constants
WIN_REWARDS    = {1: 32.0, 2: 16.0, 3: 8.0, 4: 4.0, 5: 2.0, 6: 1.0}
LOSS_REWARD    = -6.0
INFO_GAIN_COEF =  0.5


class WordleEnv:
    WORD_LEN    = 5
    MAX_GUESSES = 6
    EMPTY_LETTER = 26
    EMPTY_COLOR  = 3
    GRAY   = 0
    YELLOW = 1
    GREEN  = 2

    def __init__(self, data_dir: str = "data"):
        words_path = os.path.join(data_dir, "words.txt")
        self.words   = self._load(words_path)
        self.answers = self.words
        self.guesses = self.words
        self.vocab_size = len(self.words)
        self.obs_dim = self.WORD_LEN * self.MAX_GUESSES * 2 + 1
        print(f"[env] {self.vocab_size} words  |  obs_dim={self.obs_dim}")
        self.secret         = ""
        self.step_num       = 0
        self.board_letters  = np.array([], dtype=np.int32)
        self.board_colors   = np.array([], dtype=np.int32)
        self.done           = False
        self.history: list  = []
        self.valid_mask_arr = np.ones(self.vocab_size, dtype=bool)

    def reset(self, secret: str = None):
        if secret is not None:
            assert len(secret) == self.WORD_LEN, "Secret must be 5 letters"
            assert secret.lower() in self.answers, f"'{secret}' not in word list"
            self.secret = secret.lower()
        else:
            self.secret = random.choice(self.answers)
        self.step_num      = 0
        self.board_letters = np.full(self.WORD_LEN * self.MAX_GUESSES, self.EMPTY_LETTER, dtype=np.int32)
        self.board_colors  = np.full(self.WORD_LEN * self.MAX_GUESSES, self.EMPTY_COLOR,  dtype=np.int32)
        self.done          = False
        self.history       = []
        self.valid_mask_arr = np.ones(self.vocab_size, dtype=bool)
        return self._obs(), self.valid_mask_arr.copy()

    def step(self, action: int):
        assert not self.done, "Episode finished - call reset() first."
        assert 0 <= action < self.vocab_size

        guess  = self.guesses[action]
        colors = self._score(guess, self.secret)

        start = self.step_num * self.WORD_LEN
        for i in range(self.WORD_LEN):
            self.board_letters[start + i] = ord(guess[i]) - ord('a')
            self.board_colors [start + i] = colors[i]

        self.history.append((guess, colors))
        self.step_num += 1

        won  = all(c == self.GREEN for c in colors)
        over = won or (self.step_num >= self.MAX_GUESSES)
        self.done = over

        # Save BEFORE recomputing mask
        before = int(self.valid_mask_arr.sum())
        if not self.done:
            self._recompute_mask()
        after = int(self.valid_mask_arr.sum())

        # Information gain: how much did this guess shrink the search space?
        info_gain = np.log(before + 1) - np.log(after + 1)

        reward = INFO_GAIN_COEF * info_gain
        if over:
            reward += WIN_REWARDS[self.step_num] if won else LOSS_REWARD

        info = {
            "won":        won,
            "guess":      guess,
            "colors":     colors,
            "secret":     self.secret,
            "step":       self.step_num,
            "candidates": after,
        }
        return self._obs(), reward, self.done, info

    def valid_mask(self) -> np.ndarray:
        return self.valid_mask_arr.copy()

    def render(self):
        icons = {self.GRAY: "⬛", self.YELLOW: "🟨", self.GREEN: "🟩"}
        print(f"Secret: {self.secret}  |  Step {self.step_num}/{self.MAX_GUESSES}")
        for guess, colors in self.history:
            print("  " + guess.upper() + "  " + "".join(icons[c] for c in colors))
        print()

    def _recompute_mask(self):
        if not self.history:
            self.valid_mask_arr = np.ones(self.vocab_size, dtype=bool)
            return
        last_guess, last_colors = self.history[-1]
        new_mask = np.zeros(self.vocab_size, dtype=bool)
        for idx in np.where(self.valid_mask_arr)[0]:
            if self._score(last_guess, self.guesses[idx]) == last_colors:
                new_mask[idx] = True
        self.valid_mask_arr = new_mask if new_mask.any() else np.ones(self.vocab_size, dtype=bool)

    @staticmethod
    def _score(guess: str, secret: str) -> list:
        result = [0] * 5
        pool   = {}
        for i in range(5):
            if guess[i] == secret[i]:
                result[i] = 2
            else:
                pool[secret[i]] = pool.get(secret[i], 0) + 1
        for i in range(5):
            if result[i] != 2:
                ch = guess[i]
                if pool.get(ch, 0) > 0:
                    result[i] = 1
                    pool[ch] -= 1
        return result

    def _obs(self) -> np.ndarray:
        """Integer indices for embedding lookup — shape (61,)"""
        return np.concatenate([
            self.board_letters,
            self.board_colors,
            np.array([self.step_num], dtype=np.int32),
        ])

    @staticmethod
    def _load(path: str) -> list:
        with open(path) as f:
            return [line.strip().lower() for line in f if len(line.strip()) == 5]
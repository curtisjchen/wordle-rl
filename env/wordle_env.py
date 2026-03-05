"""
wordle_env.py — Pure Python Wordle environment (no Gymnasium required).

Observation:  float32 array of shape (obs_dim,)
    [0  :30] normalised letter indices per tile  (0–25 = a–z,  26 = empty)
    [30 :60] normalised color  per tile          (0=gray, 1=yellow, 2=green, 3=empty)
    [60]     normalised current step             (0–5)

Action:  integer index into the valid-guesses vocabulary

Reward:
    +1.0   win
    -1.0   loss
    +0.05 * #greens  every step  (shaping signal)
"""

import os
import random
import numpy as np


class WordleEnv:
    WORD_LEN    = 5
    MAX_GUESSES = 6

    # Sentinel values for unfilled tiles
    EMPTY_LETTER = 26
    EMPTY_COLOR  = 3

    # Color constants
    GRAY   = 0
    YELLOW = 1
    GREEN  = 2

    # ------------------------------------------------------------------ init
    def __init__(self, data_dir: str = "data"):
        words_path = os.path.join(data_dir, "words.txt")
        
        self.words   = self._load(words_path)
        self.answers = self.words
        self.guesses = self.words

        self.vocab_size = len(self.guesses)
        # 30 letters + 30 colors + 1 step counter
        self.obs_dim = self.WORD_LEN * self.MAX_GUESSES * 2 + 1

        print(f"[env] {len(self.answers)} answer words, "
              f"{self.vocab_size} guess words, obs_dim={self.obs_dim}")

        # Internal state (set properly in reset)
        self.secret         = ""
        self.step_num       = 0
        self.board_letters  = np.array([])
        self.board_colors   = np.array([])
        self.done           = False
        self.history: list  = []
        self.valid_mask_arr = np.ones(self.vocab_size, dtype=bool)

    # --------------------------------------------------------- public API
    def reset(self, secret: str = None):
        """
        Start a new episode.
        Returns (obs, valid_mask).
        Pass `secret` to fix the target word (e.g. from the interactive app).
        """
        if secret is not None:
            assert len(secret) == self.WORD_LEN, "Secret must be 5 letters"
            assert secret.lower() in self.answers, f"'{secret}' not in answers list"
            self.secret = secret.lower()
        else:
            self.secret = random.choice(self.answers)

        self.step_num      = 0
        self.board_letters = np.full(
            self.WORD_LEN * self.MAX_GUESSES, self.EMPTY_LETTER, dtype=np.int32
        )
        self.board_colors  = np.full(
            self.WORD_LEN * self.MAX_GUESSES, self.EMPTY_COLOR,  dtype=np.int32
        )
        self.done          = False
        self.history       = []

        # All words valid at the start — no need to iterate
        self.valid_mask_arr = np.ones(self.vocab_size, dtype=bool)

        return self._obs(), self.valid_mask_arr.copy()

    def step(self, action: int):
        """
        Apply action (word index).
        Returns (obs, reward, done, info).
        """
        assert not self.done, "Episode finished — call reset() first."
        assert 0 <= action < self.vocab_size

        guess  = self.guesses[action]
        colors = self._score(guess, self.secret)

        # Write this guess to the board
        start = self.step_num * self.WORD_LEN
        for i in range(self.WORD_LEN):
            self.board_letters[start + i] = ord(guess[i]) - ord('a')
            self.board_colors [start + i] = colors[i]

        self.history.append((guess, colors))
        self.step_num += 1

        won  = all(c == self.GREEN for c in colors)
        over = won or (self.step_num >= self.MAX_GUESSES)
        self.done = over

        # Exponential scale — each guess is worth double the next
        WIN_REWARDS = {
            1: 32.0,   # almost impossible — massive reward
            2: 16.0,   # exceptional
            3:  8.0,   # great
            4:  4.0,   # average
            5:  2.0,   # below average
            6:  1.0,   # scraped through
        }
        if over:
            if won:
                reward = WIN_REWARDS[self.step_num]
            else:
                reward = -6.0

        # Update action mask for next step
        if not self.done:
            self._recompute_mask()

        info = {
            "won":    won,
            "guess":  guess,
            "colors": colors,
            "secret": self.secret,
            "step":   self.step_num,
        }
        return self._obs(), reward, self.done, info

    def valid_mask(self) -> np.ndarray:
        """Return a copy of the current boolean action mask."""
        return self.valid_mask_arr.copy()

    def render(self):
        """Simple emoji render for debugging."""
        icons = {self.GRAY: "⬛", self.YELLOW: "🟨", self.GREEN: "🟩"}
        print(f"Secret: {self.secret}  |  Step: {self.step_num}/{self.MAX_GUESSES}")
        for guess, colors in self.history:
            print("  " + guess.upper() + "  " + "".join(icons[c] for c in colors))
        print()

    # -------------------------------------------------------- internal helpers
    def _recompute_mask(self):
        """
        Narrow the valid action set based on all feedback so far.
        Only iterates over words that were already valid (shrinks each step).
        """
        if not self.history:
            self.valid_mask_arr = np.ones(self.vocab_size, dtype=bool)
            return

        last_guess, last_colors = self.history[-1]
        new_mask = np.zeros(self.vocab_size, dtype=bool)

        # Only check candidates that were still valid after the previous step
        for idx in np.where(self.valid_mask_arr)[0]:
            candidate = self.guesses[idx]
            if self._score(last_guess, candidate) == last_colors:
                new_mask[idx] = True

        # Safeguard: if mask is empty (shouldn't happen with correct word lists),
        # fall back to all words so training doesn't crash.
        self.valid_mask_arr = new_mask if new_mask.any() else np.ones(self.vocab_size, dtype=bool)

    @staticmethod
    def _score(guess: str, secret: str) -> list:
        """
        Compute Wordle color feedback.
        Handles duplicate letters correctly.
        Returns list of GRAY / YELLOW / GREEN per position.
        """
        result   = [0] * 5
        pool     = {}          # remaining unmatched letters in secret

        # First pass — greens
        for i in range(5):
            if guess[i] == secret[i]:
                result[i] = 2  # GREEN
            else:
                pool[secret[i]] = pool.get(secret[i], 0) + 1

        # Second pass — yellows
        for i in range(5):
            if result[i] != 2:
                ch = guess[i]
                if pool.get(ch, 0) > 0:
                    result[i] = 1  # YELLOW
                    pool[ch] -= 1

        return result

    def _obs(self) -> np.ndarray:
        return np.concatenate([
            self.board_letters.astype(np.float32) / 26.0,
            self.board_colors .astype(np.float32) /  3.0,
            np.array([self.step_num / self.MAX_GUESSES], dtype=np.float32),
        ])
    @staticmethod
    def _load(path: str) -> list:
        with open(path) as f:
            return [line.strip().lower() for line in f if len(line.strip()) == 5]
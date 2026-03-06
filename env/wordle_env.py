import os
import random
import numpy as np

# --- REWARD STRUCTURE ---
WIN_REWARDS = {
    1: 1.0,
    2: 3.0,
    3: 3.5,
    4: 1.0,
    5: 0.5,
    6: -1.0
}
LOSS_REWARD    = -5.0
STEP_PENALTY   = -0.1
INFO_GAIN_COEF = 0.8  # kept for backwards compat


class WordleEnv:
    """
    Single-environment Wordle for evaluation and manual play.

    Observation format (183-dim float32) — the "knowledge state":
        [0  :26 ] letter_gray     — 1.0 if letter confirmed absent from word
        [26 :52 ] letter_present  — 1.0 if letter confirmed present (yellow or green)
        [52 :182] letter_green    — 5×26 grid, 1.0 if letter confirmed green at position p
        [182:183] step_frac       — current guess index / 6.0

    This is far more semantically useful than raw tile indices:
    the agent never has to re-derive "A is absent" from a gray tile token.
    """

    WORD_LEN    = 5
    MAX_GUESSES = 6
    GRAY        = 0
    YELLOW      = 1
    GREEN       = 2

    # Knowledge-state obs dim: 26 (gray) + 26 (present) + 5*26 (green per pos) + 1 (step)
    OBS_DIM = 26 + 26 + 5 * 26 + 1  # = 183

    def __init__(self, data_dir: str = "data"):
        words_path = os.path.join(data_dir, "words.txt")
        if not os.path.exists(words_path):
            raise FileNotFoundError(
                f"Could not find {words_path}. "
                "Please create a text file with one 5-letter word per line."
            )

        self.words      = self._load(words_path)
        self.answers    = self.words
        self.guesses    = self.words
        self.vocab_size = len(self.words)
        self.obs_dim    = self.OBS_DIM

        # State (initialised properly in reset())
        self.secret         = ""
        self.step_num       = 0
        self.done           = False
        self.history: list  = []

        # Knowledge state arrays (float32 for direct use as network input)
        self.letter_gray    = np.zeros(26, dtype=np.float32)
        self.letter_present = np.zeros(26, dtype=np.float32)
        self.letter_green   = np.zeros((self.MAX_GUESSES, 26), dtype=np.float32)  # [pos, letter]

        # Compatibility: mask over vocab indicating still-valid candidates
        self.valid_mask_arr = np.ones(self.vocab_size, dtype=bool)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def reset(self, secret: str = None):
        self.secret = secret.lower() if secret else random.choice(self.answers)

        self.step_num       = 0
        self.done           = False
        self.history        = []
        self.letter_gray[:] = 0.0
        self.letter_present[:] = 0.0
        self.letter_green[:]   = 0.0
        self.valid_mask_arr[:] = True

        return self._obs(), self.valid_mask_arr.copy()

    def step(self, action: int):
        """Standard single-step (used for manual play / eval)."""
        assert not self.done, "Call reset() before stepping after episode end."

        guess  = self.guesses[action]
        colors = self._score(guess, self.secret)

        # Update knowledge state
        for pos in range(self.WORD_LEN):
            l = ord(guess[pos]) - ord('a')
            c = colors[pos]
            if c == self.GRAY:
                self.letter_gray[l] = 1.0
            elif c == self.YELLOW:
                self.letter_present[l] = 1.0
            elif c == self.GREEN:
                self.letter_present[l] = 1.0
                self.letter_green[pos, l] = 1.0

        self.step_num += 1
        won       = all(c == self.GREEN for c in colors)
        self.done = won or (self.step_num >= self.MAX_GUESSES)
        self.history.append((guess, colors))

        reward = 0.0
        if self.done:
            reward = WIN_REWARDS.get(self.step_num, 0.0) if won else LOSS_REWARD

        return self._obs(), reward, self.done, {}

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _obs(self) -> np.ndarray:
        step_frac = np.array([self.step_num / float(self.MAX_GUESSES)], dtype=np.float32)
        return np.concatenate([
            self.letter_gray,               # 26
            self.letter_present,            # 26
            self.letter_green.flatten(),    # 5*26 = 130
            step_frac,                      # 1
        ])                                  # total = 183

    @staticmethod
    def _score(guess: str, secret: str) -> list:
        result = [0] * 5
        pool   = {}
        # Green pass
        for i in range(5):
            if guess[i] == secret[i]:
                result[i] = 2
            else:
                pool[secret[i]] = pool.get(secret[i], 0) + 1
        # Yellow pass
        for i in range(5):
            if result[i] != 2:
                ch = guess[i]
                if pool.get(ch, 0) > 0:
                    result[i] = 1
                    pool[ch] -= 1
        return result

    @staticmethod
    def _load(path: str) -> list:
        with open(path) as f:
            return [line.strip().lower() for line in f if len(line.strip()) == 5]
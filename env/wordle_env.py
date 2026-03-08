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

    Observation format (313-dim float32):
        [0  :26 ]  letter_gray       — 1.0 if letter confirmed absent
        [26 :52 ]  letter_present    — 1.0 if letter confirmed present (yellow or green)
        [52 :182]  letter_green      — 5x26 grid, confirmed GREEN at position p
        [182:312]  letter_yellow_not — 5x26 grid, present but NOT at position p
        [312:313]  step_frac         — current guess index / 6.0

    Example: guess "CRANE", secret "MOCHA"
        C -> yellow pos 0: letter_present[C]=1, letter_yellow_not[0,C]=1
        R -> gray:          letter_gray[R]=1
        A -> yellow pos 2: letter_present[A]=1, letter_yellow_not[2,A]=1
        N -> gray:          letter_gray[N]=1
        E -> gray:          letter_gray[E]=1

    Test words:
        If data/test_words.txt exists (e.g. NYT Wordle answer list), it is
        loaded and exposed as self.test_indices — indices into self.words for
        each test word. The eval env uses only these as secrets while keeping
        the full vocab as the guess action space, matching how NYT Wordle works.
        Falls back to full vocab if test_words.txt is not found.
    """

    WORD_LEN    = 5
    MAX_GUESSES = 6
    GRAY        = 0
    YELLOW      = 1
    GREEN       = 2

    # 26 (gray) + 26 (present) + 5x26 (green) + 5x26 (yellow_not) + 1 (step)
    OBS_DIM = 26 + 26 + 5 * 26 + 5 * 26 + 1  # = 313

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

        # Test word indices — secrets used only during evaluation
        test_path = os.path.join(data_dir, "test_words.txt")
        if os.path.exists(test_path):
            test_words  = self._load(test_path)
            word_to_idx = {w: i for i, w in enumerate(self.words)}
            self.test_indices = np.array(
                [word_to_idx[w] for w in test_words if w in word_to_idx],
                dtype=np.int32
            )
            print(f"Test set: {len(self.test_indices)} words loaded from {test_path}")
        else:
            self.test_indices = np.arange(self.vocab_size, dtype=np.int32)
            print(f"test_words.txt not found — eval uses full vocab ({self.vocab_size} words)")

        self.secret         = ""
        self.step_num       = 0
        self.done           = False
        self.history: list  = []

        self.letter_gray       = np.zeros(26,      dtype=np.float32)
        self.letter_present    = np.zeros(26,      dtype=np.float32)
        self.letter_green      = np.zeros((5, 26), dtype=np.float32)
        self.letter_yellow_not = np.zeros((5, 26), dtype=np.float32)

        self.valid_mask_arr = np.ones(self.vocab_size, dtype=bool)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def reset(self, secret: str = None):
        self.secret = secret.lower() if secret else random.choice(self.answers)

        self.step_num  = 0
        self.done      = False
        self.history   = []

        self.letter_gray[:]       = 0.0
        self.letter_present[:]    = 0.0
        self.letter_green[:]      = 0.0
        self.letter_yellow_not[:] = 0.0
        self.valid_mask_arr[:]    = True

        return self._obs(), self.valid_mask_arr.copy()

    def step(self, action: int):
        """Standard single-step (used for manual play / eval)."""
        assert not self.done, "Call reset() before stepping after episode end."

        guess  = self.guesses[action]
        colors = self._score(guess, self.secret)

        for pos in range(self.WORD_LEN):
            l = ord(guess[pos]) - ord('a')
            c = colors[pos]
            if c == self.GRAY:
                self.letter_gray[l] = 1.0
            elif c == self.YELLOW:
                self.letter_present[l]         = 1.0
                self.letter_yellow_not[pos, l] = 1.0
            elif c == self.GREEN:
                self.letter_present[l]    = 1.0
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
            self.letter_gray,                 # 26
            self.letter_present,              # 26
            self.letter_green.flatten(),      # 130
            self.letter_yellow_not.flatten(), # 130
            step_frac,                        # 1
        ])                                    # total = 313

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

    @staticmethod
    def _load(path: str) -> list:
        with open(path) as f:
            return [line.strip().lower() for line in f if len(line.strip()) == 5]
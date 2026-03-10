import os
import random
import numpy as np

class WordleEnv:
    """
    Refactored Wordle Environment with Letter Count Tracking.
    Observation (313-dim):
        [0   :26 ]  min_counts (normalized 0-1, representing 0-5)
        [26  :52 ]  max_counts (normalized 0-1, representing 0-5)
        [52  :182]  letter_green (5x26 grid)
        [182 :312]  letter_yellow_not (5x26 grid)
        [312 :313]  step_frac (0-1)
    """
    WORD_LEN    = 5
    MAX_GUESSES = 6
    OBS_DIM     = 313

    def __init__(self, data_dir: str = "data"):
        words_path = os.path.join(data_dir, "words.txt")
        if not os.path.exists(words_path):
            raise FileNotFoundError(f"Could not find {words_path}")

        self.words = self._load(words_path)
        self.vocab_size = len(self.words)
        
        test_path = os.path.join(data_dir, "test_words.txt")
        if os.path.exists(test_path):
            test_words = self._load(test_path)
            word_to_idx = {w: i for i, w in enumerate(self.words)}
            self.test_indices = np.array([word_to_idx[w] for w in test_words if w in word_to_idx], dtype=np.int32)
        else:
            self.test_indices = np.arange(self.vocab_size, dtype=np.int32)

        self.reset()

    def _load(self, path):
        with open(path) as f:
            return [line.strip().lower() for line in f if len(line.strip()) == 5]

    @staticmethod
    def _score(guess: str, secret: str) -> list:
        """Returns list of 5 ints: 0=Gray, 1=Yellow, 2=Green"""
        result = [0] * 5
        pool = {}
        for g, s in zip(guess, secret):
            if g != s: pool[s] = pool.get(s, 0) + 1
        for i, (g, s) in enumerate(zip(guess, secret)):
            if g == s: result[i] = 2
        for i, (g, s) in enumerate(zip(guess, secret)):
            if g != s and pool.get(g, 0) > 0:
                result[i] = 1
                pool[g] -= 1
        return result

    def reset(self, secret: str = None):
        self.secret = secret.lower() if secret else random.choice(self.words)
        self.step_num = 0
        self.done = False
        
        self.letter_min_count = np.zeros(26, dtype=np.float32)
        self.letter_max_count = np.ones(26, dtype=np.float32) * 5.0
        self.letter_green = np.zeros((5, 26), dtype=np.float32)
        self.letter_yellow_not = np.zeros((5, 26), dtype=np.float32)
        
        return self._obs()

    def _obs(self):
        return np.concatenate([
            self.letter_min_count / 5.0,
            self.letter_max_count / 5.0,
            self.letter_green.flatten(),
            self.letter_yellow_not.flatten(),
            [self.step_num / 6.0]
        ]).astype(np.float32)

    def step(self, action: int):
        guess = self.words[action]
        colors = self._score(guess, self.secret)
        
        # Track counts for duplicate letter logic
        char_stats = {} # char -> [total_guessed, colored_count]
        
        for i, (char, color) in enumerate(zip(guess, colors)):
            c_idx = ord(char) - ord('a')
            if color == 1:
                self.letter_yellow_not[i, c_idx] = 1.0
            elif color == 2:
                self.letter_green[i, c_idx] = 1.0
            
            stats = char_stats.get(char, [0, 0])
            stats[0] += 1
            if color > 0: stats[1] += 1
            char_stats[char] = stats

        for char, (total, colored) in char_stats.items():
            c_idx = ord(char) - ord('a')
            self.letter_min_count[c_idx] = max(self.letter_min_count[c_idx], colored)
            if total > colored:
                self.letter_max_count[c_idx] = colored

        self.step_num += 1
        won = all(c == 2 for c in colors)
        self.done = won or self.step_num >= self.MAX_GUESSES
        reward = 1.0 if won else ( -1.0 if self.done else 0.0)
        
        return self._obs(), reward, self.done, {"won": won}
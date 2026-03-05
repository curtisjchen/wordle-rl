import os
import random
import numpy as np

# --- AGGRESSIVE REWARD STRUCTURE ---
# Goal: 3.5 Average Guesses.
# Strategy: 
#   1. Massive penalty for losing (-20)
#   2. Small penalty for every step taken (-0.5)
#   3. Peak reward for solving in 3 steps (20.0)

WIN_REWARDS    = {
    1: 5.0,   # Luck (low reward to prevent overfitting to lucky starters)
    2: 20.0,  # Great
    3: 20.0,  # THE GOAL (Targeting 3.5 avg means hitting 3s often)
    4: 10.0,  # Par
    5: 5.0,   # Slow
    6: 1.0    # Panic
}

LOSS_REWARD    = -20.0 
STEP_PENALTY   = -0.5 
INFO_GAIN_COEF = 0.8  

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
        if not os.path.exists(words_path):
            raise FileNotFoundError(f"Could not find {words_path}. Please create a text file with 5-letter words.")
            
        self.words   = self._load(words_path)
        self.answers = self.words # Using same list for answers and guesses
        self.guesses = self.words
        self.vocab_size = len(self.words)
        
        # 61 dims: 30 letters + 30 colors + 1 step index
        self.obs_dim = self.WORD_LEN * self.MAX_GUESSES * 2 + 1
        
        self.secret          = ""
        self.step_num        = 0
        self.board_letters   = np.array([], dtype=np.int32)
        self.board_colors    = np.array([], dtype=np.int32)
        self.done            = False
        self.history: list   = []
        self.valid_mask_arr  = np.ones(self.vocab_size, dtype=bool)

    def reset(self, secret: str = None):
        if secret is not None:
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
        """
        Standard single-step logic (used for manual play/eval).
        Training uses the vectorized logic in train_cpu.py
        """
        guess = self.guesses[action]
        colors = self._score(guess, self.secret)
        
        # Update board
        start = self.step_num * 5
        for i in range(5):
            self.board_letters[start + i] = ord(guess[i]) - ord('a')
            self.board_colors [start + i] = colors[i]
            
        self.step_num += 1
        won = all(c == 2 for c in colors)
        self.done = won or (self.step_num >= 6)
        
        # Simple reward for manual play (training uses the vectorized one)
        reward = 0
        if self.done:
            reward = WIN_REWARDS.get(self.step_num, 0) if won else LOSS_REWARD
            
        return self._obs(), reward, self.done, {}

    @staticmethod
    def _score(guess: str, secret: str) -> list:
        result = [0] * 5
        pool   = {}
        # 1. Green pass
        for i in range(5):
            if guess[i] == secret[i]:
                result[i] = 2
            else:
                pool[secret[i]] = pool.get(secret[i], 0) + 1
        # 2. Yellow pass
        for i in range(5):
            if result[i] != 2:
                ch = guess[i]
                if pool.get(ch, 0) > 0:
                    result[i] = 1
                    pool[ch] -= 1
        return result

    def _obs(self) -> np.ndarray:
        return np.concatenate([
            self.board_letters,
            self.board_colors,
            np.array([self.step_num], dtype=np.int32),
        ])

    @staticmethod
    def _load(path: str) -> list:
        with open(path) as f:
            return [line.strip().lower() for line in f if len(line.strip()) == 5]
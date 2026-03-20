import sys
import os

# --- PATH SETUP ---
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)
# ------------------

import numpy as np
import time

DATA_DIR  = "data"
MODEL_DIR = "models"

from env.wordle_env import WordleEnv

def build_score_cache(env: WordleEnv) -> np.ndarray:
    """
    Builds a rectangular cache of shape (all_words x candidate_secrets).
    - Rows  = any word the agent might guess (14.5k)
    - Cols  = only candidate secret words (2.3k)
    This single cache supports both curriculum phases.
    """
    cache_path = os.path.join(DATA_DIR, "score_cache.npy")
    if os.path.exists(cache_path):
        print(" Loading score cache...", end=" ", flush=True)
        cache = np.load(cache_path)
        print(f"OK {cache.shape}")
        return cache

    all_words       = env.words                          # 14.5k
    secret_indices  = env.test_indices                   # indices into all_words for candidates
    secret_words    = [all_words[i] for i in secret_indices]  # 2.3k

    num_guesses = len(all_words)
    num_secrets = len(secret_words)
    print(f" Building score cache ({num_guesses} guesses x {num_secrets} secrets)...")

    t0    = time.time()
    cache = np.zeros((num_guesses, num_secrets), dtype=np.uint8)

    for g_idx, guess in enumerate(all_words):
        for s_idx, secret in enumerate(secret_words):
            colors  = WordleEnv._score(guess, secret)
            encoded = sum(c * (3 ** i) for i, c in enumerate(colors))
            cache[g_idx, s_idx] = encoded

    np.save(cache_path, cache)  # outside both loops
    print(f" Done in {(time.time()-t0):.1f}s.")
    return cache

def main():
    env = WordleEnv(DATA_DIR)
    build_score_cache(env)

if __name__ == "__main__":
    main()
import numpy as np
import os
import time

DATA_DIR       = "data"
MODEL_DIR      = "models"

from env.wordle_env import WordleEnv

def build_score_cache(env: WordleEnv) -> np.ndarray:
    """Pre-computes 3^5 scores for every word pair."""
    cache_path = os.path.join(DATA_DIR, "score_cache.npy")

    if os.path.exists(cache_path):
        print(" Loading score cache...", end=" ", flush=True)
        cache = np.load(cache_path)
        print(f"OK {cache.shape}")
        return cache

    # env.guesses = 14,000 allowed words
    # env.secrets = 2,700 possible answers
    num_guesses = len(env.guesses)
    num_secrets = len(env.secrets)
    
    print(f" Building score cache ({num_guesses}x{num_secrets})...")
    t0    = time.time()
    
    # Non-square matrix
    cache = np.zeros((num_guesses, num_secrets), dtype=np.uint8)

    for g_idx, guess in enumerate(env.guesses):
        for s_idx, secret in enumerate(env.secrets):
            colors  = WordleEnv._score(guess, secret)
            # Encode [2,0,1,0,0] -> scalar 0-242
            encoded = sum(c * (3 ** i) for i, c in enumerate(colors))
            cache[g_idx, s_idx] = encoded

    np.save(cache_path, cache)
    print(f" Done in {(time.time()-t0):.1f}s.")
    return cache

def main():
    env = WordleEnv(DATA_DIR)
    build_score_cache(env)

if __name__ == "__main__":
    main()
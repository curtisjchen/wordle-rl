import sys
import os
import numpy as np

# --- PATH SETUP ---
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)
# ------------------

from env.wordle_env import WordleEnv

DATA_DIR = "data"

def find_best_openers(top_n=20):
    print("Loading environment and score cache...")
    env = WordleEnv(DATA_DIR)
    score_cache = np.load(os.path.join(DATA_DIR, "score_cache.npy"))
    
    n_secrets = score_cache.shape[1]
    print(f"Vocab size: {env.vocab_size} | Candidates (secrets): {n_secrets}")
    print(f"Computing info gain for all {env.vocab_size} words...\n")

    results = []
    for g_idx in range(env.vocab_size):
        scores = score_cache[g_idx]  # shape (n_secrets,)
        unique, counts = np.unique(scores, return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-9))
        results.append((entropy, env.words[g_idx], g_idx))

    results.sort(reverse=True)

    print(f"{'Rank':<6} {'Word':<10} {'Entropy (bits)':<16}")
    print("-" * 50)
    candidate_set = set(env.test_indices.tolist())
    for rank, (entropy, word, g_idx) in enumerate(results[:top_n], 1):
        print(f"{rank:<6} {word.upper():<10} {entropy:<16.4f}")

if __name__ == "__main__":
    find_best_openers(top_n=50)
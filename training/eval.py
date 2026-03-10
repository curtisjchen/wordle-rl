import os
import torch
import numpy as np

import sys

# --- PATH SETUP ---
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# Adjust imports based on where you save this script
import env
from env.wordle_env import WordleEnv
from agent.network import WordleNetwork

# ANSI color codes for terminal printing
GREEN_BG = "\033[42m\033[30m"
YELLOW_BG = "\033[43m\033[30m"
GRAY_BG = "\033[47m\033[30m"
RESET = "\033[0m"

def format_guess(guess, scores):
    colored_chars = []
    for char, score in zip(guess, scores):
        if score == 2:
            colored_chars.append(f"{GREEN_BG} {char.upper()} {RESET}")
        elif score == 1:
            colored_chars.append(f"{YELLOW_BG} {char.upper()} {RESET}")
        else:
            colored_chars.append(f"{GRAY_BG} {char.upper()} {RESET}")
    return "".join(colored_chars)

def evaluate(model_path, num_games=100):
    device = torch.device("cpu")
    env = WordleEnv("data")
    
    # Load the trained network
    net = WordleNetwork(env.OBS_DIM, env.vocab_size).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    net.eval() # Set network to evaluation mode

    wins = 0
    total_guesses = 0
    
    # --- NEW: Evaluation Masking (Optional, but recommended) ---
    # If you want to evaluate Phase 1 (only guessing candidate words), set this to True.
    # If you are evaluating Phase 2 (full 14k actions), set this to False.
    EVALUATE_PHASE_1 = False 
    
    if EVALUATE_PHASE_1:
        eval_mask = torch.zeros(env.vocab_size, dtype=torch.bool)
        eval_mask[env.test_indices] = True
        eval_mask = eval_mask.to(device)
    else:
        eval_mask = None
    # -----------------------------------------------------------

    print(f"\nEvaluating Model: {model_path}")
    print(f"Playing {num_games} games...")

    for i in range(num_games):
        obs = env.reset()
        
        # --- THE FIX: Force the secret word to be from the candidate list ---
        # env.test_indices is a numpy array of integers (e.g., [0, 4, 15, ...])
        # We randomly select one of those indices.
        random_test_idx = np.random.choice(env.test_indices)
        
        # Manually override the environment's secret word
        env.secret_word = env.words[random_test_idx]
        # --------------------------------------------------------------------
        
        done = False
        step_count = 0
        used_actions = set() # Prevent deterministic loop traps

        print(f"\n--- Game {i+1} (Secret: {env.secret_word.upper()}) ---")

        while not done:
            with torch.no_grad():
                o_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                
                logits, _ = net(o_t, mask=eval_mask)
                
                for a in used_actions:
                    logits[0, a] = -1e8
                
                action_idx = torch.argmax(logits, dim=-1).item()
                used_actions.add(action_idx)

            guess = env.words[action_idx]
            obs, reward, done, info = env.step(action_idx) # <-- Fixed unpacking
            step_count += 1
            
            # --- THE COLOR FIX ---
            # Get the [2, 0, 1, 0, 0] score array to pass to your formatting function
            colors = WordleEnv._score(guess, env.secret_word)
            formatted_guess = format_guess(guess, colors)
            
            print(f"Step {step_count}:  {formatted_guess}")
            # ---------------------

        # --- THE TRUE WIN FIX ---
        # The game is actually won if the final guess matches the secret word perfectly
        won = (guess == env.secret_word)
        # ------------------------

        if won:
            wins += 1
            total_guesses += step_count
            print(f"Result: WON in {step_count} guesses.")
        else:
            total_guesses += 6
            print(f"Result: LOST. The word was {env.secret_word.upper()}")

    win_rate = (wins / num_games) * 100
    avg_guesses = total_guesses / num_games
    
    print("\n" + "="*30)
    print(f"EVALUATION RESULTS")
    print(f"Win Rate:    {win_rate:.1f}%")
    print(f"Avg Guesses: {avg_guesses:.2f}")
    print("="*30)

if __name__ == "__main__":
    # Point this to your best saved model!
    evaluate("models/wordle_it200.pt", num_games=10)
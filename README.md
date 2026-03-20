# Wordle RL

A reinforcement learning agent trained to play Wordle, built from scratch using pure PyTorch and PPO. Includes an interactive Pygame app where you type a secret word and watch the model guess.

![Python](https://img.shields.io/badge/python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange) ![License](https://img.shields.io/badge/license-MIT-green)

---

## Demo

The user types a 5-letter secret word. The trained agent guesses it in real time.

```
Secret word: CRANE

Guess 1: STALE  ⬛🟨⬛🟩⬛
Guess 2: CRANE  🟩🟩🟩🟩🟩   ✅ solved in 2!
```

---

## How It Works

### Environment
A pure Python Wordle environment with no external RL libraries. Each episode the agent has up to 6 guesses to identify a hidden 5-letter word. After each guess it receives color feedback (green / yellow / gray) and a reward signal.

### Reward
The reward function has three components that encourage both winning and efficient play:

| Signal | When | Purpose |
|--------|------|---------|
| `0.5 × log(before/after)` | Every step | Reward information gain — shrinking the candidate pool |
| `32 / 16 / 8 / 4 / 2 / 1` | Win in 1–6 guesses | Exponential win bonus |
| `-6.0` | Loss | Terminal penalty |

The info gain signal already implicitly penalises wasting steps — a guess that barely shrinks the candidate pool gets near-zero reward, naturally pushing the agent toward efficient play. The exponential win scale means winning in 2 guesses (16.0) is rewarded 4× more than winning in 4 (4.0), matching the actual difficulty curve of Wordle.

### Network
An embedding-based actor-critic MLP. Rather than feeding raw numbers, each board tile is represented as learned embeddings:

```
Letter embedding:  27 values → 32 dims   (a–z + empty)
Color  embedding:   4 values →  8 dims   (gray / yellow / green / empty)
Step   embedding:   7 values →  8 dims   (0–6)

30 tiles × (32 + 8) + 8 step = 1208-dim input
→ Linear(1208, 256) → ReLU
→ Linear(256,  256) → ReLU
→ Policy head: Linear(256, vocab_size)   — action logits
→ Value  head: Linear(256, 1)            — state value estimate
```

### Training
PPO (Proximal Policy Optimisation) implemented from scratch. The fast trainer runs 32 environments in parallel with a precomputed score cache that makes mask recomputation a single vectorised numpy op rather than a Python loop.

---

## Project Structure

```
wordle-rl/
│
├── data/
│   ├── fetch_words.py       # download word list (run once)
│   └── words.txt            # 14,855 valid 5-letter words (after fetch)
│
├── env/
│   └── wordle_env.py        # Wordle environment
│
├── agent/
│   ├── network.py           # embedding network (policy + value heads)
│   └── ppo.py               # PPO algorithm + rollout buffer
│
├── training/
│   ├── train.py             # reference training loop
│   └── train_fast.py        # fast training — vectorized envs + score cache
│
├── app/
│   └── main.py              # interactive Pygame app
│
├── models/
│   └── fast_final.pt        # trained model weights
│
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install torch numpy pygame
# or with uv:
uv add torch numpy pygame
```

### 2. Download word list
```bash
python data/fetch_words.py
```

### 3. Train
```bash
# Fast trainer (recommended) — builds score cache on first run (~3 min), then trains
uv run training/train_fast.py

# Reference trainer — simpler code, slower
uv run training/train.py
```

Checkpoints are saved to `models/` every N iterations.

### 4. Play
```bash
uv run app/main.py

# Use a specific checkpoint:
uv run app/main.py --model models/fast_ckpt_02500.pt
```

Type a 5-letter word in the input box and press Enter. The model will try to guess your word.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy
- Pygame (app only)
- ~500MB disk space for score cache

No GPU required — the model is small enough that CPU training converges in a few hours.

---

## Acknowledgements

Word list sourced from [dracos/valid-wordle-words](https://gist.github.com/dracos/dd0668f281e685bad51479e5acaadb93).  
PPO implementation inspired by [CleanRL](https://github.com/vwxyzjn/cleanrl).

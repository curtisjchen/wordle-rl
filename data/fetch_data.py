"""
Run this ONCE before training or playing.
Downloads the original Wordle word lists from a public GitHub mirror.

Usage:
    python data/fetch_words.py
"""

import urllib.request
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# The original Wordle answer list (~2315 words) and full valid guess list (~10657 words)
# These are widely mirrored from the original NYT Wordle source code.
SOURCES = {
    "valid_words.txt": (
        "https://gist.githubusercontent.com/dracos/dd0668f281e685bad51479e5acaadb93/raw/6bfa15d263d6d5b63840a8e5b64e04b382fdb079/valid-wordle-words.txt"
    ),
}


def fetch(url, dest):
    print(f"  Downloading {dest} ...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        with open(dest) as f:
            n = sum(1 for line in f if len(line.strip()) == 5)
        print(f"OK  ({n} words)")
    except Exception as e:
        print(f"FAILED\n  Error: {e}")
        raise


if __name__ == "__main__":
    print("Fetching Wordle word lists...\n")
    for filename, url in SOURCES.items():
        dest = os.path.join(DATA_DIR, filename)
        if os.path.exists(dest):
            with open(dest) as f:
                n = sum(1 for line in f if len(line.strip()) == 5)
            print(f"  {filename} already exists ({n} words) — skipping.")
        else:
            fetch(url, dest)
    print("\nDone. You can now run training/train.py")
import numpy as np


def load_words(path):
    with open(path) as f:
        return [w.strip().lower() for w in f if len(w.strip()) == 5]


def feedback(guess, target):

    result = [0]*5
    target_chars = list(target)

    for i in range(5):
        if guess[i] == target[i]:
            result[i] = 2
            target_chars[i] = None

    for i in range(5):
        if result[i] == 0 and guess[i] in target_chars:
            result[i] = 1
            target_chars[target_chars.index(guess[i])] = None

    return tuple(result)


def filter_candidates(candidates, guess, pattern):

    return [w for w in candidates if feedback(guess, w) == pattern]


def letter_frequency(words):

    freq = np.zeros(26)

    for w in words:
        for c in set(w):
            freq[ord(c)-97] += 1

    if len(words) > 0:
        freq /= len(words)

    return freq


def positional_frequency(words):

    pos = np.zeros((5,26))

    for w in words:
        for i,c in enumerate(w):
            pos[i,ord(c)-97] += 1

    if len(words) > 0:
        pos /= len(words)

    return pos.flatten()


def encode_state(candidates):

    size_feature = np.array([np.log(len(candidates)+1)])

    letter_freq = letter_frequency(candidates)

    pos_freq = positional_frequency(candidates)

    return np.concatenate([size_feature, letter_freq, pos_freq])
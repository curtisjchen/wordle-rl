with open('data/words.txt') as f:
    words = set(w.strip() for w in f)
with open('data/test_words.txt') as f:
    test_words = [w.strip() for w in f]

missing = [w for w in test_words if w not in words]
print(f'Test words: {len(test_words)}')
print(f'Missing from words.txt: {len(missing)}')
if missing:
    print('Missing:', missing)
else:
    print('All test words are in words.txt ✓')
import re
from collections import defaultdict, Counter
import random

class BigramModel:
    def __init__(self, corpus):
        """
        corpus: list[str]
        """
        self.bigram_counts = defaultdict(Counter)
        self.vocab = set()

        for text in corpus:
            tokens = self._tokenize(text)
            self.vocab.update(tokens)
            for w1, w2 in zip(tokens, tokens[1:]):
                self.bigram_counts[w1][w2] += 1

        # fallback distribution if a token is unseen
        self._all_tokens = sorted(list(self.vocab)) if self.vocab else ["the"]

    def _tokenize(self, text):
        text = text.lower().strip()
        # simple word tokenizer
        return re.findall(r"[a-z']+", text)

    def generate_text(self, start_word, length):
        """
        start_word: str
        length: int (number of additional words to generate)
        """
        if length <= 0:
            return start_word

        start = (start_word or "").lower().strip()
        tokens = self._tokenize(start)
        if not tokens:
            tokens = [random.choice(self._all_tokens)]

        current = tokens[-1]
        out = tokens[:]

        for _ in range(length):
            next_candidates = self.bigram_counts.get(current)
            if not next_candidates:
                nxt = random.choice(self._all_tokens)
            else:
                # sample proportional to counts
                words, counts = zip(*next_candidates.items())
                total = sum(counts)
                r = random.randint(1, total)
                cum = 0
                nxt = words[-1]
                for w, c in zip(words, counts):
                    cum += c
                    if r <= cum:
                        nxt = w
                        break

            out.append(nxt)
            current = nxt

        return " ".join(out)

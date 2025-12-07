from pathlib import Path
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_RNN_WEIGHTS = Path("models/rnn_lm.pt")

def _tokenize(text):
    return re.findall(r"[a-z']+", text.lower())

class TinyLSTMLM(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden_dim=128, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden

class RNNTextGenerator:
    """
    Minimal word-level LSTM generator.

    If weights exist at models/rnn_lm.pt, we load them.
    Otherwise we fall back to a tiny model initialized from a small default corpus.
    """

    def __init__(self, weights_path=DEFAULT_RNN_WEIGHTS):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # A tiny default corpus just to build a vocab
        self.corpus = [
            "this is a small recurrent model for text generation",
            "module seven adds lstm to the text api",
            "we keep endpoints stable for grading",
        ]

        self.vocab = self._build_vocab(self.corpus)
        self.itos = {i: w for w, i in self.vocab.items()}
        self.stoi = self.vocab

        self.model = TinyLSTMLM(len(self.vocab)).to(self.device)

        if Path(weights_path).exists():
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)

        self.model.eval()

    def _build_vocab(self, texts):
        words = []
        for t in texts:
            words.extend(_tokenize(t))
        uniq = sorted(set(words))
        # add special tokens
        vocab = {"<unk>": 0}
        for w in uniq:
            if w not in vocab:
                vocab[w] = len(vocab)
        return vocab

    def _encode(self, words):
        return [self.stoi.get(w, 0) for w in words]

    def _decode(self, ids):
        return [self.itos.get(i, "<unk>") for i in ids]

    @torch.no_grad()
    def generate(self, start_word, length):
        """
        start_word: str
        length: int (# words to generate)
        """
        if length <= 0:
            return start_word or ""

        seed_words = _tokenize(start_word or "")
        if not seed_words:
            seed_words = ["module"]

        ids = self._encode(seed_words)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)

        hidden = None
        # warm up the hidden state on seed
        _, hidden = self.model(x, hidden)

        last_id = x[0, -1].view(1, 1)

        out_ids = ids[:]
        for _ in range(length):
            logits, hidden = self.model(last_id, hidden)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            nid = int(next_id.item())
            out_ids.append(nid)
            last_id = next_id

        out_words = self._decode(out_ids)
        return " ".join(out_words)

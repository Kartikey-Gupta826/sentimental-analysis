"""Command-line demo: type a review, get the predicted sentiment.
Usage:  python3 predict.py "the movie was not very boring"
        python3 predict.py            (interactive mode)
Works with the vocabulary of the training corpus; unknown words map to <unk>."""
import json, sys
import numpy as np
import data
from nn import SeqClassifier

def load():
    w = np.load("models/best.npz", allow_pickle=True)
    vocab = json.load(open("models/vocab.json"))
    net = SeqClassifier(str(w["cell"]), len(vocab), int(w["emb"]), int(w["hidden"]))
    net.set_weights({k: w[k] for k in ("E", "W", "U", "b", "Wo", "bo")})
    return net, vocab, int(w["maxlen"]), str(w["cell"])

def predict(net, vocab, maxlen, text):
    X, _, _ = data.encode([{"text": text, "label": 0}], vocab, maxlen)
    return float(net.predict_proba(X)[0])

if __name__ == "__main__":
    net, vocab, maxlen, cell = load()
    def show(t):
        p = predict(net, vocab, maxlen, t)
        print(f"[{cell.upper()}] {'POSITIVE' if p >= .5 else 'NEGATIVE'}  (P(positive) = {p:.3f})  <- {t}")
    if len(sys.argv) > 1: show(" ".join(sys.argv[1:]))
    else:
        while True:
            t = input("review> ").strip()
            if not t: break
            show(t)

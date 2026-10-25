"""Runs every experiment of the report and writes results.json, figures/*.png and models/best.npz.
Usage:  python3 run_experiment.py                      (synthetic corpus, 3 seeds per model)
        python3 run_experiment.py --csv "IMDB Dataset.csv" --limit 5000 --maxlen 200   (real IMDb, untested here)"""
import argparse, json, os, time
import numpy as np
import data
from nn import SeqClassifier, gradient_check, GATES
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

NAMES = {"rnn": "SimpleRNN", "lstm": "LSTM", "gru": "GRU"}

def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int); y = y.astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec, "f1": f1, "confusion": [[tn, fp], [fn, tp]]}

def bce(y, p):
    e = 1e-12
    return float(-np.mean(y * np.log(p + e) + (1 - y) * np.log(1 - p + e)))

def fit(net, Xtr, ytr, Xva, yva, epochs, patience, lr, batch, seed):
    rng = np.random.default_rng(1000 + seed)
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_seconds": []}
    best, best_w, bad = 1e9, None, 0
    for ep in range(epochs):
        t0 = time.time()
        order = rng.permutation(len(Xtr))
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            _, g = net.loss_and_grads(Xtr[idx], ytr[idx])
            net.step(g, lr=lr)
        ptr, pva = net.predict_proba(Xtr), net.predict_proba(Xva)
        hist["epoch_seconds"].append(time.time() - t0)
        hist["train_loss"].append(bce(ytr, ptr)); hist["val_loss"].append(bce(yva, pva))
        hist["train_acc"].append(float(((ptr >= .5) == ytr).mean())); hist["val_acc"].append(float(((pva >= .5) == yva).mean()))
        if hist["val_loss"][-1] < best - 1e-4:
            best, best_w, bad = hist["val_loss"][-1], net.get_weights(), 0
            hist["best_epoch"] = ep + 1
        else:
            bad += 1
            if bad >= patience: break
    net.set_weights(best_w)                                  # early stopping: restore best-validation weights
    hist["epochs_run"] = len(hist["val_loss"])
    return hist

def group_acc(recs, correct, key):
    out = {}
    for k in sorted({r[key] for r in recs}):
        m = np.array([r[key] == k for r in recs])
        out[str(k)] = float(correct[m].mean())
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--emb", type=int, default=32); ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--maxlen", type=int, default=64); ap.add_argument("--csv", default=None)
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--out", default="results.json")
    a = ap.parse_args()
    os.makedirs("figures", exist_ok=True); os.makedirs("models", exist_ok=True)
    R = {"config": vars(a)}

    print("gradient check ...", flush=True)
    R["gradient_check"] = {c: gradient_check(c) for c in GATES}
    print(R["gradient_check"], flush=True)

    recs = data.load_csv(a.csv, limit=a.limit) if a.csv else data.generate(a.n, seed=0)
    tr, va, te = data.split(recs, seed=0)
    vocab = data.build_vocab(tr)
    Xtr, ytr, t1 = data.encode(tr, vocab, a.maxlen); Xva, yva, t2 = data.encode(va, vocab, a.maxlen); Xte, yte, t3 = data.encode(te, vocab, a.maxlen)
    lens = [len(data.tokenize(r["text"])) for r in recs]
    R["data"] = {"reviews": len(recs), "train": len(tr), "val": len(va), "test": len(te), "vocab_size": len(vocab),
                 "maxlen": a.maxlen, "truncated_reviews": t1 + t2 + t3, "mean_tokens": float(np.mean(lens)), "max_tokens": int(max(lens)),
                 "positive_fraction_train": float(ytr.mean()),
                 "label_noise_observed": float(np.mean([r.get("label") != r.get("clean_label", r.get("label")) for r in recs])),
                 "test_category_counts": {k: sum(r["category"] == k for r in te) for k in sorted({r["category"] for r in te})}}
    print(R["data"], flush=True)
    yte_clean = np.array([r.get("clean_label", r["label"]) for r in te], dtype=float)

    # ---- baselines (same split)
    R["baselines"] = {"majority_class": {"accuracy": float(max(yte.mean(), 1 - yte.mean()))}}
    for name, ngram in (("tfidf_logreg_unigram", (1, 1)), ("tfidf_logreg_bigram", (1, 2))):
        vec = TfidfVectorizer(tokenizer=data.tokenize, token_pattern=None, lowercase=False, ngram_range=ngram)
        clf = LogisticRegression(max_iter=2000, C=10.0).fit(vec.fit_transform([r["text"] for r in tr]), ytr)
        p = clf.predict_proba(vec.transform([r["text"] for r in te]))[:, 1]
        m = metrics(yte, p); corr = ((p >= .5) == yte)
        m["accuracy_vs_clean_labels"] = float(((p >= .5) == yte_clean).mean())
        m["by_category"] = group_acc(te, corr, "category"); m["by_trailing"] = group_acc(te, corr, "n_trailing")
        R["baselines"][name] = m
        print(name, round(m["accuracy"], 4), m["by_category"], flush=True)

    # ---- sequence models
    R["runs"] = {c: [] for c in GATES}; keep = {}
    for cell in GATES:
        for s in range(a.seeds):
            net = SeqClassifier(cell, len(vocab), a.emb, a.hidden, seed=s)
            t0 = time.time()
            h = fit(net, Xtr, ytr, Xva, yva, a.epochs, a.patience, a.lr, a.batch, s)
            pte = net.predict_proba(Xte); m = metrics(yte, pte); corr = ((pte >= .5) == yte)
            m.update({"seed": s, "params": net.n_params(), "train_seconds": time.time() - t0,
                      "best_epoch": h["best_epoch"], "epochs_run": h["epochs_run"], "val_acc_best": h["val_acc"][h["best_epoch"] - 1],
                      "train_acc_best": h["train_acc"][h["best_epoch"] - 1], "test_loss": bce(yte, pte),
                      "accuracy_vs_clean_labels": float(((pte >= .5) == yte_clean).mean()),
                      "by_category": group_acc(te, corr, "category"), "by_trailing": group_acc(te, corr, "n_trailing"),
                      "mean_epoch_seconds": float(np.mean(h["epoch_seconds"]))})
            if s == 0: m["history"] = h
            R["runs"][cell].append(m); keep[(cell, s)] = (net, h, m)
            print(f"{NAMES[cell]} seed{s}: test acc {m['accuracy']:.4f} f1 {m['f1']:.4f} best_ep {m['best_epoch']} "
                  f"params {m['params']} {m['train_seconds']:.0f}s", flush=True)
            json.dump(R, open(a.out, "w"), indent=1)

    # ---- summary
    R["summary"] = {}
    for cell, runs in R["runs"].items():
        f = lambda k: [r[k] for r in runs]
        R["summary"][cell] = {k: {"mean": float(np.mean(f(k))), "std": float(np.std(f(k)))} for k in
                              ("accuracy", "precision", "recall", "f1", "val_acc_best", "train_acc_best", "best_epoch", "train_seconds", "mean_epoch_seconds", "accuracy_vs_clean_labels")}
        R["summary"][cell]["params"] = runs[0]["params"]
        for grp in ("by_category", "by_trailing"):
            R["summary"][cell][grp] = {k: float(np.mean([r[grp][k] for r in runs])) for k in runs[0][grp]}
        R["summary"][cell]["confusion_seed0"] = runs[0]["confusion"]

    # ---- learning-rate sensitivity (seed 0): same data, same schedule, three learning rates per model
    R["lr_sensitivity"] = {}
    for cell in GATES:
        R["lr_sensitivity"][cell] = {}
        for lr in (1e-3, 3e-3, 1e-2):
            net = SeqClassifier(cell, len(vocab), a.emb, a.hidden, seed=0)
            h = fit(net, Xtr, ytr, Xva, yva, a.epochs, a.patience, lr, a.batch, 0)
            R["lr_sensitivity"][cell][str(lr)] = {"val_acc": h["val_acc"][h["best_epoch"] - 1], "best_epoch": h["best_epoch"], "epochs_run": h["epochs_run"]}
            print("lr sweep", cell, lr, R["lr_sensitivity"][cell][str(lr)], flush=True)
        json.dump(R, open(a.out, "w"), indent=1)

    # ---- embedding analysis on the best-validation run
    (bc, bs) = max(keep, key=lambda k: keep[k][2]["val_acc_best"])
    bnet = keep[(bc, bs)][0]; E = bnet.P["E"]
    def vecs(words): return np.array([E[vocab[w]] for w in words if w in vocab])
    def unit(M): return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    P_, N_ = unit(vecs(data.POS)), unit(vecs(data.NEG))
    within = lambda M: float((M @ M.T)[~np.eye(len(M), dtype=bool)].mean())
    R["embedding"] = {"model": f"{NAMES[bc]} seed {bs}", "cos_within_positive": within(P_), "cos_within_negative": within(N_),
                      "cos_between": float((P_ @ N_.T).mean())}
    print(R["embedding"], flush=True)
    np.savez("models/best.npz", cell=bc, emb=a.emb, hidden=a.hidden, maxlen=a.maxlen, **bnet.P)
    json.dump(vocab, open("models/vocab.json", "w"))
    R["best_model"] = {"cell": bc, "seed": bs}
    json.dump(R, open(a.out, "w"), indent=1)

    # ---- figures
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    col = {"rnn": "#d95f02", "lstm": "#1b9e77", "gru": "#7570b3"}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for c in GATES:
        h = R["runs"][c][0]["history"]; ep = np.arange(1, len(h["val_loss"]) + 1)
        ax[0].plot(ep, h["train_acc"], color=col[c], label=f"{NAMES[c]} train"); ax[0].plot(ep, h["val_acc"], "--", color=col[c], label=f"{NAMES[c]} val")
        ax[1].plot(ep, h["train_loss"], color=col[c], label=f"{NAMES[c]} train"); ax[1].plot(ep, h["val_loss"], "--", color=col[c], label=f"{NAMES[c]} val")
    ax[0].set(xlabel="Epoch", ylabel="Accuracy", title="Training and validation accuracy"); ax[1].set(xlabel="Epoch", ylabel="Binary cross-entropy", title="Training and validation loss")
    ax[0].legend(fontsize=7, ncol=2); ax[1].legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig("figures/curves.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(10, 3.3))
    for i, c in enumerate(GATES):
        cm = np.array(R["runs"][c][0]["confusion"]); ax[i].imshow(cm, cmap="Blues")
        for (r_, c_), v in np.ndenumerate(cm): ax[i].text(c_, r_, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
        ax[i].set(title=NAMES[c], xticks=[0, 1], yticks=[0, 1], xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"], xlabel="Predicted", ylabel="Actual")
    fig.tight_layout(); fig.savefig("figures/confusion.png", dpi=150); plt.close(fig)

    cats = list(R["data"]["test_category_counts"]); w = 0.16
    series = [("Bag-of-words (unigram)", [R["baselines"]["tfidf_logreg_unigram"]["by_category"][k] for k in cats], "#999999"),
              ("TF-IDF + bigrams", [R["baselines"]["tfidf_logreg_bigram"]["by_category"][k] for k in cats], "#cccccc")]
    series += [(NAMES[c], [R["summary"][c]["by_category"][k] for k in cats], col[c]) for c in GATES]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for j, (n, v, cl) in enumerate(series): ax.bar(np.arange(len(cats)) + (j - 2) * w, v, w, label=n, color=cl)
    ax.set(xticks=range(len(cats)), xticklabels=cats, ylabel="Test accuracy", ylim=(0.4, 1.02), title="Accuracy by review type"); ax.legend(fontsize=7, ncol=3)
    fig.tight_layout(); fig.savefig("figures/by_category.png", dpi=150); plt.close(fig)

    tr_keys = sorted(R["summary"]["lstm"]["by_trailing"], key=int)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for c in GATES: ax.plot(tr_keys, [R["summary"][c]["by_trailing"][k] for k in tr_keys], "o-", color=col[c], label=NAMES[c])
    ax.plot(tr_keys, [R["baselines"]["tfidf_logreg_bigram"]["by_trailing"][k] for k in tr_keys], "s--", color="#999999", label="TF-IDF + bigrams")
    ax.set(xlabel="Neutral sentences after the verdict", ylabel="Test accuracy", title="Effect of distance between verdict and end of text"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig("figures/trailing.png", dpi=150); plt.close(fig)

    words = [w_ for w_ in data.POS + data.NEG if w_ in vocab]; M = np.array([E[vocab[w_]] for w_ in words]); Mc = M - M.mean(0)
    U_, S_, _ = np.linalg.svd(Mc, full_matrices=False); Z = U_[:, :2] * S_[:2]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for i, w_ in enumerate(words):
        pos = w_ in data.POS; ax.scatter(Z[i, 0], Z[i, 1], color="#1b9e77" if pos else "#d95f02", s=14); ax.annotate(w_, Z[i], fontsize=7, xytext=(2, 2), textcoords="offset points")
    ax.set(title=f"Learned word embeddings (PCA, {R['embedding']['model']}); green = positive words, orange = negative words", xlabel="PC1", ylabel="PC2")
    ax.title.set_fontsize(8); fig.tight_layout(); fig.savefig("figures/embeddings.png", dpi=150); plt.close(fig)
    print("ALL DONE", flush=True)

if __name__ == "__main__":
    main()

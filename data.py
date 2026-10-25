"""Data module: synthetic movie-review generator, tokenizer, vocabulary, padding, optional IMDb CSV loader.

WHY SYNTHETIC: the build environment had no internet, so the real IMDb dataset could not be downloaded.
The generator makes reviews whose label depends on WORD ORDER (negation, 'but' contrast, decoy opinions),
so a model that ignores order (bag of words) is expected to fail on part of the data.
For real IMDb data use load_csv() (same pipeline, same models)."""
import csv, random, re
from collections import Counter
import numpy as np

POS = ["excellent", "wonderful", "brilliant", "enjoyable", "fantastic", "superb", "delightful", "charming",
       "gripping", "moving", "hilarious", "stunning", "engaging", "memorable", "outstanding", "touching"]
NEG = ["boring", "terrible", "awful", "disappointing", "dull", "dreadful", "weak", "tedious",
       "forgettable", "painful", "messy", "predictable", "lifeless", "annoying", "clumsy", "pointless"]
# (past form, base form): "i loved it" / "i did not love it"
LOVE = [("loved", "love"), ("enjoyed", "enjoy"), ("adored", "adore"), ("liked", "like")]
HATE = [("hated", "hate"), ("disliked", "dislike"), ("despised", "despise"), ("loathed", "loathe")]
INTENS = ["", "", "very", "really", "truly", "quite", "absolutely"]
SUBJ = ["the movie", "the film", "the story", "the acting", "the ending", "the script", "the soundtrack",
        "the direction", "the plot", "the cast", "the pacing", "the dialogue"]
VERB = ["was", "is"]
OBJ = ["it", "this movie", "the film", "this story"]
FILLER = [
    "i watched it on a rainy saturday with my friends",
    "the theatre was almost full that night",
    "it runs for about two hours",
    "my brother recommended it to me last week",
    "we bought popcorn before the show started",
    "the director has made several other films",
    "it was released earlier this year",
    "the lead actor is quite famous",
    "i went to see it after work",
    "the screening started a little late",
    "there were many people in the queue",
    "it is based on a novel from the nineties",
]
CATEGORIES = ["simple", "negation", "contrast", "decoy"]

def _clause(rng, polarity, negated):
    """One evaluative clause whose overall polarity is `polarity` (1 positive, 0 negative).
    If negated, the word polarity is flipped and 'not' is inserted, so the overall polarity is unchanged."""
    word_pol = polarity ^ 1 if negated else polarity
    if rng.random() < 0.7:                                    # adjective clause: "the movie was not very boring"
        pool = POS if word_pol else NEG
        adj = rng.choice(pool)
        inten = rng.choice(INTENS)
        parts = [rng.choice(SUBJ), rng.choice(VERB)] + (["not"] if negated else []) + ([inten] if inten else []) + [adj]
        if not negated and rng.random() < 0.3:                # coordination: "the movie was excellent and enjoyable"
            parts += ["and", rng.choice([w for w in pool if w != adj])]
    else:                                                     # verb clause: "i loved it" / "i did not love it"
        past, base = rng.choice(LOVE if word_pol else HATE)
        parts = ["i"] + (["did", "not", base] if negated else [past]) + [rng.choice(OBJ)]
    return " ".join(parts)

def make_review(rng, label_noise):
    cat = rng.choice(CATEGORIES)
    final = rng.randint(0, 1)                                 # polarity of the verdict (true label)
    neg_p = 0.3
    if cat == "simple":
        core = _clause(rng, final, False)
    elif cat == "negation":
        core = _clause(rng, final, True)
    elif cat == "contrast":                                   # "A but B" -> B decides
        a = _clause(rng, rng.randint(0, 1), rng.random() < neg_p)
        b = _clause(rng, final, rng.random() < neg_p)
        core = f"{a} , {rng.choice(['but', 'but', 'however'])} {b}"
    else:                                                     # decoy opinion earlier, verdict last
        a = _clause(rng, rng.randint(0, 1), rng.random() < neg_p)
        core = f"{a} . {_clause(rng, final, rng.random() < neg_p)}"
    n_lead, n_trail = rng.randint(0, 2), rng.randint(0, 3)
    sents = [rng.choice(FILLER) for _ in range(n_lead)] + [core] + [rng.choice(FILLER) for _ in range(n_trail)]
    text = " . ".join(sents) + " ."
    label = final ^ 1 if rng.random() < label_noise else final   # annotation noise
    return {"text": text, "label": label, "clean_label": final, "category": cat, "n_trailing": n_trail}

def generate(n=6000, seed=0, label_noise=0.03):
    rng = random.Random(seed)
    return [make_review(rng, label_noise) for _ in range(n)]

def tokenize(text):
    return re.findall(r"[a-z']+|[.,!?]", text.lower())

def split(records, seed=0, frac=(0.7, 0.15, 0.15)):
    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)
    n_tr, n_va = int(frac[0] * len(idx)), int(frac[1] * len(idx))
    pick = lambda ids: [records[i] for i in ids]
    return pick(idx[:n_tr]), pick(idx[n_tr:n_tr + n_va]), pick(idx[n_tr + n_va:])

def build_vocab(train_records, max_vocab=None, min_freq=1):
    """Vocabulary from TRAINING data only (no leakage). 0 = <pad>, 1 = <unk>."""
    c = Counter(t for r in train_records for t in tokenize(r["text"]))
    words = [w for w, f in c.most_common(max_vocab) if f >= min_freq]
    return {w: i + 2 for i, w in enumerate(words)} | {"<pad>": 0, "<unk>": 1}

def encode(records, vocab, maxlen):
    """Tokens -> ids; pre-pad with 0 and truncate from the beginning (keeps the end, like Keras defaults)."""
    X = np.zeros((len(records), maxlen), dtype=np.int64)
    truncated = 0
    for i, r in enumerate(records):
        ids = [vocab.get(t, 1) for t in tokenize(r["text"])]
        if len(ids) > maxlen: truncated += 1; ids = ids[-maxlen:]
        if ids: X[i, -len(ids):] = ids
    y = np.array([r["label"] for r in records], dtype=np.float64)
    return X, y, truncated

def load_csv(path, text_col="review", label_col="sentiment", limit=None, seed=0):
    """Real-data loader, e.g. the Kaggle 'IMDB Dataset.csv' (review, sentiment=positive|negative). NOT tested here."""
    recs = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lab = row[label_col].strip().lower()
            recs.append({"text": re.sub(r"<[^>]+>", " ", row[text_col]), "label": 1 if lab in ("positive", "1", "pos") else 0,
                         "category": "real", "n_trailing": 0})
    random.Random(seed).shuffle(recs)
    return recs[:limit] if limit else recs

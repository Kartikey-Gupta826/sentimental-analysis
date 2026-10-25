"""SimpleRNN / LSTM / GRU sentiment classifier written from scratch in NumPy (forward pass, back-propagation
through time, Adam, gradient clipping). No deep-learning framework was available in the build environment.

Architecture: token ids -> Embedding(V, d) -> recurrent layer (hidden H, last hidden state) -> Dense(1) -> sigmoid.
Loss: binary cross-entropy."""
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

GATES = {"rnn": 1, "lstm": 4, "gru": 3}

class SeqClassifier:
    def __init__(self, cell, vocab_size, emb_dim=32, hidden=32, seed=0):
        assert cell in GATES
        self.cell, self.V, self.d, self.H = cell, vocab_size, emb_dim, hidden
        rng = np.random.default_rng(seed)
        G, H, d = GATES[cell], hidden, emb_dim
        E = rng.normal(0, 0.1, (vocab_size, d)); E[0] = 0.0                 # row 0 = <pad>, fixed at zero
        lim = np.sqrt(6.0 / (d + H))
        W = rng.uniform(-lim, lim, (d, G * H))
        U = np.concatenate([self._orthogonal(rng, H) for _ in range(G)], axis=1)
        b = np.zeros(G * H)
        if cell == "lstm": b[H:2 * H] = 1.0                                  # forget-gate bias = 1
        lo = np.sqrt(6.0 / (H + 1))
        self.P = {"E": E, "W": W, "U": U, "b": b, "Wo": rng.uniform(-lo, lo, (H, 1)), "bo": np.zeros(1)}
        self.m = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.t = 0

    @staticmethod
    def _orthogonal(rng, n):
        q, r = np.linalg.qr(rng.normal(size=(n, n)))
        return q * np.sign(np.diag(r))

    def n_params(self):
        return int(sum(v.size for v in self.P.values()))

    # ---------------------------------------------------------------- forward
    def forward(self, X, keep=True):
        P, H = self.P, self.H
        B, T = X.shape
        emb = P["E"][X]                                   # (B, T, d)
        h = np.zeros((B, H)); c = np.zeros((B, H))
        caches = []
        for t in range(T):
            x = emb[:, t, :]
            if self.cell == "rnn":
                hn = np.tanh(x @ P["W"] + h @ P["U"] + P["b"])
                cache = (x, h, hn)
            elif self.cell == "lstm":
                a = x @ P["W"] + h @ P["U"] + P["b"]
                i, f = sigmoid(a[:, :H]), sigmoid(a[:, H:2 * H])
                g, o = np.tanh(a[:, 2 * H:3 * H]), sigmoid(a[:, 3 * H:])
                cn = f * c + i * g
                tc = np.tanh(cn)
                hn = o * tc
                cache = (x, h, c, i, f, g, o, tc)
                c = cn
            else:                                          # GRU: h_new = (1-z)*h + z*h_tilde
                ax = x @ P["W"] + P["b"]
                U = P["U"]
                zr = h @ U[:, :2 * H]
                z = sigmoid(ax[:, :H] + zr[:, :H])
                r = sigmoid(ax[:, H:2 * H] + zr[:, H:])
                ht = np.tanh(ax[:, 2 * H:] + (r * h) @ U[:, 2 * H:])
                hn = (1 - z) * h + z * ht
                cache = (x, h, z, r, ht)
            if keep: caches.append(cache)
            h = hn
        p = sigmoid(h @ P["Wo"] + P["bo"]).ravel()
        return p, (X, h, caches) if keep else None

    # ---------------------------------------------------------------- loss + backward (BPTT)
    def loss_and_grads(self, X, y):
        P, H = self.P, self.H
        B, T = X.shape
        p, (X, hT, caches) = self.forward(X, keep=True)
        eps = 1e-12
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        dz = ((p - y) / B)[:, None]                       # d loss / d logit
        grads = {k: np.zeros_like(v) for k, v in P.items()}
        grads["Wo"] = hT.T @ dz; grads["bo"] = dz.sum(0)
        dh = dz @ P["Wo"].T
        dc = np.zeros((B, H))
        dX = np.zeros((B, T, self.d))
        for t in reversed(range(T)):
            cache = caches[t]
            if self.cell == "rnn":
                x, hp, hn = cache
                da = dh * (1 - hn ** 2)
                grads["W"] += x.T @ da; grads["U"] += hp.T @ da; grads["b"] += da.sum(0)
                dX[:, t] = da @ P["W"].T; dh = da @ P["U"].T
            elif self.cell == "lstm":
                x, hp, cp, i, f, g, o, tc = cache
                do = dh * tc
                dct = dc + dh * o * (1 - tc ** 2)
                di, dg, df = dct * g, dct * i, dct * cp
                dc = dct * f
                da = np.concatenate([di * i * (1 - i), df * f * (1 - f), dg * (1 - g ** 2), do * o * (1 - o)], axis=1)
                grads["W"] += x.T @ da; grads["U"] += hp.T @ da; grads["b"] += da.sum(0)
                dX[:, t] = da @ P["W"].T; dh = da @ P["U"].T
            else:
                x, hp, z, r, ht = cache
                U = P["U"]
                dzg = dh * (ht - hp)
                dht = dh * z
                dhp = dh * (1 - z)
                da_h = dht * (1 - ht ** 2)
                drh = da_h @ U[:, 2 * H:].T
                grads["U"][:, 2 * H:] += (r * hp).T @ da_h
                dr = drh * hp
                dhp = dhp + drh * r
                da_z, da_r = dzg * z * (1 - z), dr * r * (1 - r)
                da_zr = np.concatenate([da_z, da_r], axis=1)
                grads["U"][:, :2 * H] += hp.T @ da_zr
                dhp = dhp + da_zr @ U[:, :2 * H].T
                da = np.concatenate([da_z, da_r, da_h], axis=1)
                grads["W"] += x.T @ da; grads["b"] += da.sum(0)
                dX[:, t] = da @ P["W"].T
                dh = dhp
        np.add.at(grads["E"], X, dX)
        grads["E"][0] = 0.0
        return loss, grads

    # ---------------------------------------------------------------- optimiser
    def step(self, grads, lr=3e-3, clip=5.0, b1=0.9, b2=0.999):
        gnorm = np.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
        scale = min(1.0, clip / (gnorm + 1e-12))           # global-norm gradient clipping
        self.t += 1
        for k, g in grads.items():
            g = g * scale
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t); vh = self.v[k] / (1 - b2 ** self.t)
            self.P[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
        self.P["E"][0] = 0.0
        return gnorm

    def predict_proba(self, X, batch=256):
        return np.concatenate([self.forward(X[i:i + batch], keep=False)[0] for i in range(0, len(X), batch)])

    def get_weights(self): return {k: v.copy() for k, v in self.P.items()}
    def set_weights(self, w): self.P = {k: v.copy() for k, v in w.items()}


def gradient_check(cell, seed=0, eps=1e-5):
    """Compare analytic BPTT gradients with central finite differences on a tiny network.
    Returns the maximum relative error over ALL parameters."""
    rng = np.random.default_rng(seed)
    net = SeqClassifier(cell, vocab_size=9, emb_dim=4, hidden=5, seed=seed)
    for k in net.P:                                        # make the check non-trivial (non-zero biases etc.)
        if k != "E": net.P[k] += rng.normal(0, 0.2, net.P[k].shape)
    net.P["E"][1:] = rng.normal(0, 0.5, net.P["E"][1:].shape)
    X = rng.integers(0, 9, size=(4, 7)); y = rng.integers(0, 2, size=4).astype(float)
    _, grads = net.loss_and_grads(X, y)
    worst = 0.0
    for k, arr in net.P.items():
        num = np.zeros_like(arr)
        it = np.nditer(arr, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            if k == "E" and idx[0] == 0: continue          # pad row is fixed at zero by design
            old = arr[idx]
            arr[idx] = old + eps; lp, _ = net.loss_and_grads(X, y)
            arr[idx] = old - eps; lm, _ = net.loss_and_grads(X, y)
            arr[idx] = old
            num[idx] = (lp - lm) / (2 * eps)
        a = grads[k].copy()
        if k == "E": a[0] = 0; num[0] = 0
        rel = np.abs(a - num) / np.maximum(1e-8, np.abs(a) + np.abs(num))
        worst = max(worst, float(rel.max()))
    return worst

if __name__ == "__main__":
    for c in GATES: print(c, "max relative gradient error:", gradient_check(c))

"""
MLP from scratch with NumPy.
Covers: neuron (W,b), activations (ReLU/Sigmoid/Tanh), BCE / softmax-CrossEntropy,
forward + backward propagation, SGD / Adam, L2 regularization, inverted dropout.
"""
import numpy as np

# ---------- activations (function, derivative wrt pre-activation) ----------
def relu(z):     return np.maximum(0, z)
def relu_d(z):   return (z > 0).astype(z.dtype)
def sigmoid(z):  return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
def sigmoid_d(z):
    s = sigmoid(z); return s * (1 - s)
def tanh(z):     return np.tanh(z)
def tanh_d(z):   return 1 - np.tanh(z) ** 2
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)          # numerical stability
    e = np.exp(z); return e / e.sum(axis=1, keepdims=True)

ACT = {"relu": (relu, relu_d), "sigmoid": (sigmoid, sigmoid_d), "tanh": (tanh, tanh_d)}


class NumpyMLP:
    def __init__(self, layer_sizes, activation="relu", lr=1e-2, optimizer="adam",
                 l2=0.0, dropout=0.0, seed=0):
        """layer_sizes e.g. [30, 16, 8, 1]. Last size 1 => binary (sigmoid+BCE),
        >1 => multi-class (softmax+CE)."""
        self.sizes, self.lr, self.opt, self.l2, self.p_drop = layer_sizes, lr, optimizer, l2, dropout
        self.act, self.act_d = ACT[activation]
        self.binary = layer_sizes[-1] == 1
        self.rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            scale = np.sqrt(2.0 / n_in) if activation == "relu" else np.sqrt(1.0 / n_in)  # He / Xavier
            self.W.append(self.rng.normal(0, scale, (n_in, n_out)))
            self.b.append(np.zeros((1, n_out)))
        # Adam state
        self.mW = [np.zeros_like(w) for w in self.W]; self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]; self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    # ---------------- forward ----------------
    def forward(self, X, train=False):
        self.cache = {"A": [X], "Z": [], "mask": []}
        A = X
        for i in range(len(self.W)):
            Z = A @ self.W[i] + self.b[i]
            last = i == len(self.W) - 1
            if last:
                A = sigmoid(Z) if self.binary else softmax(Z)
                mask = None
            else:
                A = self.act(Z)
                if train and self.p_drop > 0:                       # inverted dropout
                    mask = (self.rng.random(A.shape) > self.p_drop) / (1 - self.p_drop)
                    A = A * mask
                else:
                    mask = None
            self.cache["Z"].append(Z); self.cache["A"].append(A); self.cache["mask"].append(mask)
        return A

    # ---------------- loss ----------------
    def loss(self, probs, y):
        eps = 1e-12; m = len(y)
        if self.binary:
            p = np.clip(probs.ravel(), eps, 1 - eps)
            data = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        else:
            data = -np.mean(np.log(np.clip(probs[np.arange(m), y], eps, 1)))
        reg = 0.5 * self.l2 * sum((w ** 2).sum() for w in self.W) / m
        return data + reg

    # ---------------- backward ----------------
    def backward(self, y):
        m = len(y); L = len(self.W)
        if self.binary:
            dZ = (self.cache["A"][-1] - y.reshape(-1, 1)) / m           # dL/dZ for sigmoid+BCE
        else:
            dZ = self.cache["A"][-1].copy(); dZ[np.arange(m), y] -= 1; dZ /= m   # softmax+CE
        dW, db = [None] * L, [None] * L
        for i in reversed(range(L)):
            dW[i] = self.cache["A"][i].T @ dZ + self.l2 * self.W[i] / m   # + L2 gradient
            db[i] = dZ.sum(axis=0, keepdims=True)
            if i > 0:
                dA = dZ @ self.W[i].T
                if self.cache["mask"][i - 1] is not None:
                    dA = dA * self.cache["mask"][i - 1]
                dZ = dA * self.act_d(self.cache["Z"][i - 1])
        return dW, db

    # ---------------- optimizer step ----------------
    def step(self, dW, db, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for i in range(len(self.W)):
            if self.opt == "sgd":
                self.W[i] -= self.lr * dW[i]; self.b[i] -= self.lr * db[i]
            else:  # adam
                for p, g, m_, v_ in ((self.W, dW, self.mW, self.vW), (self.b, db, self.mb, self.vb)):
                    m_[i] = b1 * m_[i] + (1 - b1) * g[i]
                    v_[i] = b2 * v_[i] + (1 - b2) * g[i] ** 2
                    mh, vh = m_[i] / (1 - b1 ** self.t), v_[i] / (1 - b2 ** self.t)
                    p[i] -= self.lr * mh / (np.sqrt(vh) + eps)

    # ---------------- API ----------------
    def predict_proba(self, X): return self.forward(X, train=False)
    def predict(self, X):
        p = self.predict_proba(X)
        return (p.ravel() > 0.5).astype(int) if self.binary else p.argmax(axis=1)

    def fit(self, X, y, X_val, y_val, epochs=200, batch_size=32, verbose=True):
        hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        n = len(X)
        for ep in range(1, epochs + 1):
            idx = self.rng.permutation(n)
            for s in range(0, n, batch_size):
                b = idx[s:s + batch_size]
                self.forward(X[b], train=True)
                dW, db = self.backward(y[b])
                self.step(dW, db)
            hist["train_loss"].append(self.loss(self.predict_proba(X), y))
            hist["val_loss"].append(self.loss(self.predict_proba(X_val), y_val))
            hist["train_acc"].append((self.predict(X) == y).mean())
            hist["val_acc"].append((self.predict(X_val) == y_val).mean())
            if verbose and (ep % max(1, epochs // 10) == 0 or ep == 1):
                print(f"[NumPy] ep {ep:4d} | loss {hist['train_loss'][-1]:.4f} / {hist['val_loss'][-1]:.4f}"
                      f" | acc {hist['train_acc'][-1]:.3f} / {hist['val_acc'][-1]:.3f}")
        return hist


def gradient_check(seed=0):
    """Compare analytic gradients with numerical ones (sanity test for backprop)."""
    rng = np.random.default_rng(seed)
    for sizes, binary in (([5, 4, 1], True), ([5, 4, 3], False)):
        X = rng.normal(size=(8, 5)); y = rng.integers(0, 2 if binary else 3, 8)
        net = NumpyMLP(sizes, activation="tanh", l2=0.0, seed=seed)
        net.forward(X); dW, _ = net.backward(y)
        num = np.zeros_like(net.W[0]); h = 1e-5
        for i in range(num.shape[0]):
            for j in range(num.shape[1]):
                old = net.W[0][i, j]
                net.W[0][i, j] = old + h; lp = net.loss(net.forward(X), y)
                net.W[0][i, j] = old - h; lm = net.loss(net.forward(X), y)
                net.W[0][i, j] = old; num[i, j] = (lp - lm) / (2 * h)
        err = np.abs(num - dW[0]).max()
        print(f"Gradient check ({'binary' if binary else 'multiclass'}): max abs diff = {err:.2e}")
        assert err < 1e-6

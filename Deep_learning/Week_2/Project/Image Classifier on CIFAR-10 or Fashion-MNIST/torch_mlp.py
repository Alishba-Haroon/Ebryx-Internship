"""Same MLP re-implemented in PyTorch, with Dropout, weight decay (L2), early stopping."""
import copy
import numpy as np
import torch
import torch.nn as nn


class TorchMLP(nn.Module):
    def __init__(self, n_in, n_out, hidden=(32, 16), dropout=0.0):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_out))        # raw logits
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_torch(X_tr, y_tr, X_va, y_va, n_classes, hidden=(32, 16), dropout=0.0,
                weight_decay=0.0, lr=1e-2, epochs=200, batch_size=32, patience=None, seed=0):
    """n_classes == 2 -> single logit + BCEWithLogits; otherwise CrossEntropy."""
    torch.manual_seed(seed)
    binary = n_classes == 2
    model = TorchMLP(X_tr.shape[1], 1 if binary else n_classes, hidden, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss()

    Xt = torch.tensor(X_tr, dtype=torch.float32); Xv = torch.tensor(X_va, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32 if binary else torch.long)
    yv = torch.tensor(y_va, dtype=torch.float32 if binary else torch.long)
    fix = (lambda o: o.squeeze(1)) if binary else (lambda o: o)
    acc = (lambda o, y: ((o > 0).float() == y).float().mean().item()) if binary \
        else (lambda o, y: (o.argmax(1) == y).float().mean().item())

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for s in range(0, len(Xt), batch_size):
            b = perm[s:s + batch_size]
            opt.zero_grad()
            loss = crit(fix(model(Xt[b])), yt[b])
            loss.backward()
            opt.step()
        model.eval()                                   # turns dropout OFF for evaluation
        with torch.no_grad():
            ot, ov = fix(model(Xt)), fix(model(Xv))
            lt, lv = crit(ot, yt).item(), crit(ov, yv).item()
        hist["train_loss"].append(lt); hist["val_loss"].append(lv)
        hist["train_acc"].append(acc(ot, yt)); hist["val_acc"].append(acc(ov, yv))
        if patience:
            if lv < best - 1e-4:
                best, best_state, bad = lv, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


@torch.no_grad()
def predict_torch(model, X, binary):
    model.eval()
    out = model(torch.tensor(X, dtype=torch.float32))
    return (out.squeeze(1) > 0).long().numpy() if binary else out.argmax(1).numpy()

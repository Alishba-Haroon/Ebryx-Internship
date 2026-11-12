"""
Phase 1 project: Binary & multi-class classifiers on tabular data.

Usage:
    python run_project.py --dataset iris      # multi-class
    python run_project.py --dataset cancer    # binary
    python run_project.py --dataset titanic   # binary (downloads from OpenML)
    python run_project.py --dataset cancer --skip-torch   # NumPy only
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

from data import load_dataset, make_splits
from numpy_mlp import NumpyMLP, gradient_check

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="iris", choices=["iris", "cancer", "titanic"])
p.add_argument("--epochs", type=int, default=200)
p.add_argument("--skip-torch", action="store_true")
args = p.parse_args()
os.makedirs("figures", exist_ok=True)

X, y, class_names, task = load_dataset(args.dataset)
X_tr, X_va, X_te, y_tr, y_va, y_te = make_splits(X, y)
n_classes = len(class_names)
print(f"\nDataset={args.dataset} task={task} train/val/test={len(X_tr)}/{len(X_va)}/{len(X_te)}"
      f" features={X_tr.shape[1]}")


def plot_hist(histories: dict, title, fname):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for name, h in histories.items():
        ax[0].plot(h["train_loss"], "--", alpha=.6); ax[0].plot(h["val_loss"], label=name)
        ax[1].plot(h["val_acc"], label=name)
    ax[0].set(title=f"{title}: loss (dashed=train, solid=val)", xlabel="epoch", ylabel="loss")
    ax[1].set(title=f"{title}: validation accuracy", xlabel="epoch", ylabel="accuracy")
    ax[0].legend(); ax[1].legend(); plt.tight_layout()
    plt.savefig(f"figures/{fname}", dpi=130); plt.close()


# ------------------------------------------------------------------ 1) NumPy
print("\n=== 1) NumPy from scratch ===")
gradient_check()
out_dim = 1 if task == "binary" else n_classes
sizes = [X_tr.shape[1], 32, 16, out_dim]

np_runs = {
    "baseline":      dict(l2=0.0, dropout=0.0),
    "L2=1.0":        dict(l2=1.0, dropout=0.0),
    "dropout=0.3":   dict(l2=0.0, dropout=0.3),
    "L2+dropout":    dict(l2=1.0, dropout=0.3),
}
np_hist = {}
for name, kw in np_runs.items():
    print(f"\n-- {name}")
    net = NumpyMLP(sizes, activation="relu", lr=1e-2, optimizer="adam", seed=0, **kw)
    np_hist[name] = net.fit(X_tr, y_tr, X_va, y_va, epochs=args.epochs, verbose=(name == "baseline"))
    print(f"   test acc = {accuracy_score(y_te, net.predict(X_te)):.4f}")
plot_hist(np_hist, f"NumPy MLP ({args.dataset})", f"{args.dataset}_numpy_curves.png")

# optimizer / activation comparison (SGD vs Adam, ReLU vs Tanh vs Sigmoid)
print("\n-- Optimizer / activation comparison (val loss after training)")
cmp = {}
for opt, lr in (("sgd", 0.1), ("adam", 0.01)):
    for act in ("relu", "tanh", "sigmoid"):
        n = NumpyMLP(sizes, activation=act, optimizer=opt, lr=lr, seed=0)
        cmp[f"{opt}-{act}"] = n.fit(X_tr, y_tr, X_va, y_va, epochs=100, verbose=False)
        print(f"   {opt:4s} {act:7s} val_loss={cmp[f'{opt}-{act}']['val_loss'][-1]:.4f}"
              f" val_acc={cmp[f'{opt}-{act}']['val_acc'][-1]:.3f}")
plot_hist(cmp, "Optimizer/activation", f"{args.dataset}_opt_act.png")

# ------------------------------------------------------------------ 2) PyTorch
if not args.skip_torch:
    from torch_mlp import train_torch, predict_torch
    print("\n=== 2) PyTorch re-implementation ===")
    configs = {
        "baseline":        dict(dropout=0.0, weight_decay=0.0),
        "dropout=0.3":     dict(dropout=0.3, weight_decay=0.0),
        "weight_decay":    dict(dropout=0.0, weight_decay=1e-2),
        "dropout+wd+ES":   dict(dropout=0.3, weight_decay=1e-3, patience=20),
    }
    t_hist, models = {}, {}
    for name, kw in configs.items():
        m, h = train_torch(X_tr, y_tr, X_va, y_va, n_classes, epochs=args.epochs, **kw)
        t_hist[name], models[name] = h, m
        acc = accuracy_score(y_te, predict_torch(m, X_te, task == "binary"))
        print(f"   {name:15s} epochs_run={len(h['val_loss']):3d} test acc={acc:.4f}")
    plot_hist(t_hist, f"PyTorch MLP ({args.dataset})", f"{args.dataset}_torch_curves.png")

    best = models["dropout+wd+ES"]
    pred = predict_torch(best, X_te, task == "binary")
else:
    pred = net.predict(X_te)       # last NumPy model (L2+dropout)

print("\n=== Final evaluation on TEST set ===")
print(classification_report(y_te, pred, target_names=[str(c) for c in class_names]))
ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=class_names, cmap="Blues")
plt.title(f"Confusion matrix ({args.dataset})"); plt.savefig(f"figures/{args.dataset}_confusion.png", dpi=130)
print("Figures saved in ./figures/")

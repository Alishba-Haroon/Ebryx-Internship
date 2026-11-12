# Phase 1 Project — MLP Classifier on Tabular Data

**Datasets:** Iris (3-class), Breast Cancer (binary), Titanic (binary)

## Run
```bash
pip install -r requirements.txt
python run_project.py --dataset iris
python run_project.py --dataset cancer
python run_project.py --dataset titanic
```

## Files
| File | What it does |
|---|---|
| `data.py` | Load data, stratified 60/20/20 split, standardize (fit on train only) |
| `numpy_mlp.py` | MLP from scratch: forward/backward, ReLU/Sigmoid/Tanh, BCE/CE, SGD/Adam, L2, dropout, gradient check |
| `torch_mlp.py` | Same network in PyTorch + Dropout, weight decay, early stopping |
| `run_project.py` | Runs experiments, prints metrics, saves plots to `figures/` |

## Experiments
1. Baseline vs L2 vs Dropout vs both (overfitting control)
2. SGD vs Adam, ReLU vs Tanh vs Sigmoid
3. NumPy vs PyTorch results
4. Test-set classification report + confusion matrix

## Report questions to answer
- Where does validation loss start rising above train loss (overfitting)?
- Which regularization helped most on each dataset?
- Why does Adam converge faster than plain SGD here?
- Why is dropout switched off at evaluation (`model.eval()`)?

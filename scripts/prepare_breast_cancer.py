"""Write data/breast_cancer.csv (numeric features + integer label, label last).
Uses the dataset bundled with scikit-learn (no download)."""
import os
from sklearn.datasets import load_breast_cancer

os.makedirs("data", exist_ok=True)
d = load_breast_cancer()
X, y = d.data, d.target
with open("data/breast_cancer.csv", "w") as f:
    for row, label in zip(X, y):
        f.write(",".join(f"{v:.6g}" for v in row) + f",{int(label)}\n")
print(f"wrote data/breast_cancer.csv  shape={X.shape}  classes={sorted(set(y.tolist()))}")

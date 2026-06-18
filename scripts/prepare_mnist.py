"""Write data/mnist_train_bin.csv and data/mnist_test_bin.csv.
Pixels are binarized (pixel > 75, i.e. ~0.3*255 -> 1), 784 features + label.
Tries Keras first, falls back to OpenML via scikit-learn (needs network once)."""
import os
os.makedirs("data", exist_ok=True)
try:
    from tensorflow.keras.datasets import mnist
    (xtr, ytr), (xte, yte) = mnist.load_data()
    xtr = xtr.reshape(len(xtr), -1); xte = xte.reshape(len(xte), -1)
except Exception:
    from sklearn.datasets import fetch_openml
    mn = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mn.data.astype("float32"); y = mn.target.astype(int)
    xtr, xte, ytr, yte = X[:60000], X[60000:], y[:60000], y[60000:]

def write(path, X, y, thr=75):
    with open(path, "w") as f:
        for row, label in zip(X, y):
            bits = (row > thr).astype(int)
            f.write(",".join(map(str, bits.tolist())) + f",{int(label)}\n")

write("data/mnist_train_bin.csv", xtr, ytr)
write("data/mnist_test_bin.csv", xte, yte)
print(f"wrote binarized MNIST: train={len(xtr)} test={len(xte)} features=784")

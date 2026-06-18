"""Write data/imdb_train.txt and data/imdb_test.txt in sparse binary format:
each line is `label idx1 idx2 ...` (indices of present vocabulary words).
Requires Keras (tensorflow). N_FEATURES must match examples/imdb.rs."""
import os
os.makedirs("data", exist_ok=True)
from tensorflow.keras.datasets import imdb

N_FEATURES = 5000
(xtr, ytr), (xte, yte) = imdb.load_data(num_words=N_FEATURES)

def write(path, X, y):
    with open(path, "w") as f:
        for seq, label in zip(X, y):
            idxs = sorted({i for i in seq if 0 <= i < N_FEATURES})
            f.write(str(int(label)) + "".join(f" {i}" for i in idxs) + "\n")

write("data/imdb_train.txt", xtr, ytr)
write("data/imdb_test.txt", xte, yte)
print(f"wrote IMDb sparse BoW: train={len(xtr)} test={len(xte)} N_FEATURES={N_FEATURES}")

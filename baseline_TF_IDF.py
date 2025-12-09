import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import pathlib

DATA_DIR = pathlib.Path("data/final")

train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df  = pd.read_csv(DATA_DIR / "test.csv")

vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        max_features=50_000,
        min_df=2)
X_train = vectorizer.fit_transform(train_df.text)
X_test  = vectorizer.transform(test_df.text)


clf = LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1)
clf.fit(X_train, train_df.label)

pred = clf.predict(X_test)
acc = accuracy_score(test_df.label, pred)
f1  = f1_score(test_df.label, pred, average="macro")
κ   = cohen_kappa_score(test_df.label, pred)

print(f"Acc: {acc:.4f}")
print(f"F1 : {f1:.4f}")
print(f"κ  : {κ:.4f}")
import pandas as pd, pathlib, os
from sklearn.model_selection import train_test_split


in_dir  = pathlib.Path("data/clean")
out_dir = pathlib.Path("data/final")
os.makedirs(out_dir, exist_ok=True)


if (in_dir / "train_clean.csv").exists():
    for split in ["train", "val", "test"]:
        df = pd.read_csv(in_dir / f"{split}_clean.csv")
        df.to_csv(out_dir / f"{split}.csv", index=False)
        print(f"{split}.csv  → {df.shape[0]} 条")

elif (in_dir / "all_clean.csv").exists():
    df = pd.read_csv(in_dir / "all_clean.csv")
    train, test = train_test_split(df, test_size=0.2, stratify=df.label, random_state=42)
    train, val  = train_test_split(train, test_size=0.1, stratify=train.label, random_state=42)
    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(  out_dir / "val.csv",   index=False)
    test.to_csv( out_dir / "test.csv",  index=False)
    print("划分完成：", train.shape, val.shape, test.shape)

else:
    raise FileNotFoundError("请在 data/clean 放入 train_clean.csv / val_clean.csv / test_clean.csv")
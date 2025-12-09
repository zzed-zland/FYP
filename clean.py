import pandas as pd, pathlib, re
from tqdm import tqdm

def clean_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text if len(text) > 20 else None

root  = pathlib.Path("THUCNews")
out   = pathlib.Path("data/clean")
out.mkdir(parents=True, exist_ok=True)

files = {"cnews.train.txt": "train_clean.csv",
         "cnews.val.txt":   "val_clean.csv",
         "cnews.test.txt":  "test_clean.csv"}

for in_name, out_name in files.items():
    in_file = root / in_name
    df = pd.read_table(in_file, sep='\t', header=None, names=['label', 'text'])
    df['text'] = df['text'].astype(str).apply(clean_text)
    df = df[df['text'].notnull()]
    df.to_csv(out / out_name, index=False, encoding='utf-8')
    print(f"{in_file} → {out / out_name}  {df.shape[0]} 条")
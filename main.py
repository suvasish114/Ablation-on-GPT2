import pandas as pd
from data_cleaning import clean_text

df = pd.read_parquet("/Users/papai/Documents/NAI3/dataset/wikitext2/wikitext-2-v1/train-00000-of-00001.parquet")
for sentence in list(df.head()["text"]):
    # print(sentence)
    print(clean_text(sentence))

import pandas as pd 
import numpy as np 
import os

batch_dir = "translations"
batch_files = sorted([f for f in os.listdir(batch_dir) if f.startswith("translated_batch")])

for i, batch_file in enumerate(batch_files, 1):
    batch_path = os.path.join(batch_dir, batch_file)
    df = pd.read_csv(batch_path)

    columns = ["yor_Latn", "ibo_Latn", "hau_Latn"]

    ff = df[columns].copy()

    df.drop(columns=columns, inplace=True)

    new_df = pd.DataFrame(columns= ["combined", "language", "queue"])

    for column in columns:
        new_df["combined"] = ff[[column]]
        new_df["language"] = column.split("_")[0]
        new_df["queue"] = df["queue"]
        df = pd.concat([df, new_df], axis=0, ignore_index=True)
    
    df.to_csv(f"translations/v2/translated_batch{i}_v2.csv", index=False)
    
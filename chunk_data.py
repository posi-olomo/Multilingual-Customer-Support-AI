import pandas as pd 

df = pd.read_csv("data/aa_dataset-tickets-multi-lang-5-2-50-version.csv")

# Copy the English rows to a new DataFrame
df_eng = df[df.language == "en"].copy()

# Fill NaN values in 'subject' column with empty strings
df_eng['subject'] = df_eng['subject'].fillna('')
df_eng['combined'] = df_eng['subject'] + ' [SEP] ' + df_eng['body']
df_final = df_eng[['combined']]

batch_size = 100 
for i, chunk in enumerate(range(0, len(df), batch_size)):
    df.iloc[chunk: chunk+batch_size].to_csv(f"data/batch{i}.csv", index = False)

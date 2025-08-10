import pandas as pd 

df = pd.read_csv("data/aa_dataset-tickets-multi-lang-5-2-50-version.csv")

# Copy the English rows to a new DataFrame
df_eng = df[df.language == "en"].copy()

# Fill NaN values in 'subject' column with empty strings
df_chunk = pd.DataFrame()

# Create a dataframe with an equal number of tags, only if subject is not empty
for i in df_eng.queue.unique():
    temp = df_eng[(df_eng.queue == i) & (df_eng.subject.notnull()) & (df_eng.subject != '')].sample(n=50, random_state=42)
    df_chunk = pd.concat([df_chunk,temp], ignore_index = True)

df_chunk['combined'] = df_chunk['subject'] + ' [SEP] ' + df_chunk['body']
df_final = df_chunk[['combined','language', 'queue']]

batch_size = 100 
for i, chunk in enumerate(range(0, len(df_final), batch_size)):
    df_final.iloc[chunk: chunk+batch_size].to_csv(f"data/batch{i}.csv", index = False)

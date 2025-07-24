import pandas as pd 
import boto3
import os
from io import StringIO

# AWS Configuration
bucket_name = 'ticket-data-multilang'
# The path to the CSV file in the S3 bucket
s3_key = 'aa_dataset-tickets-multi-lang-5-2-50-version.csv'
region = 'eu-north-1'

# Initialize S3 client
s3 = boto3.client('s3', region_name=region)

# Download the CSV file from S3
print("Downloading CSV file from S3...")
response = s3.get_object(Bucket=bucket_name, Key=s3_key)
# Decode the byte files into a string
csv_content = response['Body'].read().decode('utf-8')

# Load the CSV content into a DataFrame
df = pd.read_csv(StringIO(csv_content))
print("Loaded CSV file into DataFrame.")

# Copy the English rows to a new DataFrame
df_eng = df[df.language == "en"].copy()

# Fill NaN values in 'subject' column with empty strings
df_eng['subject'] = df_eng['subject'].fillna('')
df_eng['combined'] = df_eng['subject'] + ' [SEP] ' + df_eng['body']
df_final = df_eng[['combined', 'queue']]

os.makedirs('data', exist_ok=True)

batch_size = 100 
for i, chunk in enumerate(range(0, len(df_final), batch_size)):
    file_path = f"data/batch{i}.csv"
    df_final.iloc[chunk: chunk+batch_size].to_csv(file_path, index = False)
    print(f"Saved batch to {file_path}")

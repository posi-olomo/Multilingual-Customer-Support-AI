import os
import logging 
import pandas as pd
import boto3
import openai 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------------
# ✅ Setup Logging
# ---------------------------------

logging.basicConfig(
    level = logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers = [ logging.StreamHandler()
    ]
)

# ---------------------------------
# 📥 Load and preprocess the Ticket Data
# ---------------------------------
try:
    logging.info("Reading input CSV file...")
    df = pd.read_csv("data/aa_dataset-tickets-multi-lang-5-2-50-version.csv")
    df_eng = df[df.language == "en"]
    logging.info(f"Loaded {len(df_eng)} rows from the CSV file.")
    # Fill NaN values in 'subject' column with empty strings
    df_eng['subject'] = df_eng['subject'].fillna('')

    df_eng['combined'] = df_eng['subject'] + ' [SEP] ' + df_eng['body']
    logging.info("Combined 'subject' and 'body' into 'combined' column.")
except Exception as e:
    logging.error("Failed to read the CSV file.", exc_info=True)
    raise

# ---------------------------------
# 🌐 Translate to Multiple Languages
# ---------------------------------
languages = ["yor_Latn", "ibo_Latn", "hau_Latn"]

logging.info("Loading the tokenizer and model.")
# Load model directly
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
logging.info("Finished loading the tokenizer.")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
logging.info("Finished loading the model.")

# A function to translate text
def translate(text, target_lang):
    inputs = tokenizer(text, return_tensors="pt")
    translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang))
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens = True)[0]

for lang in languages:
    try:
        logging.info(f"Translating to {lang}...")
        # Convert to Yoruba, Igbo and Hausa
        df_eng[lang] = df_eng["combined"].apply(lambda x: translate(x, lang))
        logging.info(f"Translation to {lang} completed.")

    except Exception as e:
        logging.error(f"Translation failed for {lang}", exc_info=True)
        raise

# ---------------------------------
# 💾 Save Translated CSV
# ---------------------------------
filename = "translated_tickets.csv"
try: 
    df.to_csv(filename, index = False)
    logging.info(f"Saved translated data to {filename}.")
except Exception as e:
    logging.error("Failed to save the translated CSV file.", exc_info=True)
    raise
        
# ---------------------------------
# ☁️ Upload to S3
# ---------------------------------

try: 
    logging.info("Uploading to S3...")

    session = boto3.session.Session(
        aws_access_key_id = os.environ.get['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key = os.environ.get['AWS_SECRET_ACCESS_KEY'],
        region_name = os.environ.get['AWS_REGION']
    )
    s3 = session.resource('s3')
    bucket_name = os.environ.get['S3_BUCKET_NAME']
    s3_path = f"translations/{filename}"
    s3.Bucket(bucket_name).upload_file(filename, s3_path)

    logging.info(f"File uploaded to s3://{bucket_name}/{s3_path}")
except Exception as e:
    logging.error("Failed to upload the file to S3.", exc_info=True)
    raise
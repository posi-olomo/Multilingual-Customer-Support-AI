import os
import subprocess
import logging 
import boto3
from botocore.exceptions import ClientError 

# Setup logging 
logging.basicConfig(
    level=logging.INFO, 
    format = "%(asctime)s [%(levelname)s] %(message)s",
    handlers = [logging.StreamHandler()])

def check_file_in_S3(batch_file):
    """Check if translated file already exists in S3."""
    try: 
        session = boto3.session.Session(
            aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name = os.environ.get('AWS_REGION')
        )
        s3 = session.client('s3')
        bucket_name = os.environ.get('S3_BUCKET_NAME')

        if not bucket_name:
            logging.warning("S3_BUCKET_NAME not set, will rely on local file check only")
            return False 
        
        # Construct S3 key for translated file
        translated_filename = f"translated_{batch_file}"
        s3_key = f"translations/{translated_filename}"

        # Check if file exists in S3
        s3.head_object(Bucket=bucket_name, Key=s3_key)
        logging.info(f"{batch_file} already processed (found in S3: {s3_key})")
        return True
    
    except ClientError as e: # This is a specific AWS S3 error
        error_code = e.response['Error']['Code']

        if error_code == '404': # Expected S3 error
            # File doesn't exist in S3. We haven't processed this file yet so go ahead and process it.
            return False 
        
        else: # Unexpected S3 errors
            logging.warning(f"Error checking S3 for {batch_file}: {e}")
            # Assume file doesn't exist and go ahead with processing
            return False  
        
    except Exception as e: # Non-S3 errors
        logging.warning(f"Could not check S3 for {batch_file}: {e}")
        return False

def get_s3_processed_summary():
    """Get summary of what's already been processed in S3"""
    try:
        session = boto3.session.Session(
            aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name = os.environ.get('AWS_REGION')
        )
        s3 = session.client('s3')
        bucet_name = os.environ.get('S3_BUCKET_NAME')

        if not bucket_name:
            return []

        # Get a list of processed files in S3
        response = s3.list_objects_v2(
            Bucket=bucket_name,
            Prefix=f"translations/translated_batch"
        )

        processed_files = []
        if 'Contents' in response:
            for obj in response['Contents']:
                filename = obj['Key'].split('/')[-1] # Get filename from S3 key
                original_batch = filename.replace('translated_', '') # Remove prefix
                processed_files.append(original_batch)

        return sorted(processed_files)
    except Exception as e:
        logging.warning(f"Could not get S3 summary: {e}")
        return []

# Directory containing batch files 
batch_dir = "data"
batch_files = sorted([f for f in os.listdir(batch_dir) if f.startswith("batch")])

if not batch_files:
    logging.info(f"No files found in {batch_dir}/ starting with 'batch'")
    exit(1)

logging.info(f"Found {len(batch_files)} batch files to process.")

# Create the output directory if it doesn't exist
output_dir = "translations"
os.makedirs(output_dir, exist_ok=True)

# Get summary of what's already been processed in S3
processed_in_s3 = get_s3_processed_summary()
if processed_in_s3:
    logging.info(f"Found {len(processed_in_s3)} processed files in S3:")
    for f in processed_in_s3:
        logging.info(f)

else:
    logging.info(f"Did not find any previously processed files in S3")

# Track processing stats
skipped_local = 0
skipped_s3 = 0
processed = 0
failed = 0

# Loop through each batch file
for i, batch_file in enumerate(batch_files, 1):
    batch_path = os.path.join(batch_dir, batch_file)
    logging.info(f"Processing {i}/{len(batch_Files)}: {batch_file}")

    # Define the output file name 
    output_file = os.path.join(output_dir, f"translated_{batch_file}")
    
    # Primary check: Check if it already exists in S3
    if check_file_in_s3(batch_file):
        skipped_s3 += 1
        continue
    
    # Fallback: Check if local output file already exists 
    if os.path.exists(output_file):
        logging.info(f"Skipping {batch_File}, local output already exists")
        skipped_local += 1
        continue

    try:
        logging.info(f"Starting translation for {batch_file}...")
        result = subprocess.run(
            ["python", "convert_db.py", batch_path],
            check = True,
            capture_output=True,
            text=True
        )

        # Log any output from convert_db.py
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if line.strip():  # Skip empty lines
                    logging.info(f"convert_db: {line}")
        
        logging.info(f"Successfully processed {batch_file}")
        processed += 1

    except subprocess.CalledProcessError as e:
        logging.error(f"Failed processing {batch_file}: {e}")
        if e.stdout:
            logging.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            logging.error(f"STDERR: {e.stderr}")
        failed += 1
        continue
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        break

# Final Summary
logging.info("=" * 50)
logging.info("BATCH PROCESSING COMPLETE")
logging.info("Summary:")
logging.info(f" - Total files found: {len(batch_files)}")
logging.info(f" - Successfully processed: {processed}")
logging.info(f" - Skipped (already in S3): {skipped_s3}")
logging.info(f" - Skipped (local file exists): {skipped_local}")
logging.info(f" - Failed {failed}")
logging.info("=" * 50)

# Show current S3 status
final_s3_files = get_s3_processed_summary()
logging.info(f"Total files now in S3: {len(final_s3_files)}")

if failed > 0:
    logging.warning(f"{failed} files failed to process. Check logs above for details")
    exit(1)
else:
    logging.info("All processing completed successfully.")


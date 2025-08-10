import os
import subprocess 

# Directory containing batch files 
batch_dir = "data"
batch_files = sorted([f for f in os.listdir(batch_dir) if f.startswith("batch")])

# Create the output directory if it doesn't exist
output_dir = "translations"
os.makedirs(output_dir, exist_ok=True)

# Loop through each batch file

for batch_file in batch_files:
    batch_path = os.path.join(batch_dir, batch_file)
    print(f"Processing{batch_path}...")

    # Define the output file name 
    output_file = os.path.join(output_dir, f"translated_{batch_file}")
    # Trigger the translation script with the batch file as input
    
    # Skip if the output file already exists
    if os.path.exists(output_file):
        print(f"Skipping {batch_file}, output already exists.")
        continue

    try:
        subprocess.run(["python", "convert_db.py", batch_path], check=True)

    except subprocess.CalledProcessError as e:
        print(f"Failed on {batch_file}: {e}")
        continue

    print(f"Finished {batch_path}")
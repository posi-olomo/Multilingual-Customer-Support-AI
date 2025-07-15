import os
import glob

batch_dir = "data"
pattern = os.path.join(batch_dir, "batch*.csv")

files_to_delete = glob.glob(pattern)

for file in files_to_delete:
    try:
        os.remove(file)
        print(f"Deleted: {file}")
    except Exception as e:
        print(f"Failed to delete {file}: {e}")
batch_files = sorted([f for f in os.listdir(batch_dir) if f.startswith("batch")])
import pandas as pd
import shutil
import os
from pathlib import Path
from tqdm import tqdm

# Configuration
CSV_PATH = '/home/quangnhvn34/data/AutoMask-Refinery/review_details.csv'
# The user specified this is the root containing subfolders
SOURCE_ROOT = '/home/quangnhvn34/data/fsoft/02_Working/logo_20260314/Quang_Part2_0314'
FAILED_ROOT = '/home/quangnhvn34/data/AutoMask-Refinery/Quang_Part2_0314_failed'
PASS_ROOT = '/home/quangnhvn34/data/AutoMask-Refinery/Quang_Part2_0314_pass'

def organize_images():
    """
    Organizes images, JSON, and XML files into 'pass' and 'fail' folders
    based on the review results stored in the CSV.
    """
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        return

    # Read the CSV
    print(f"Reading CSV from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # Ensure root folders exist
    os.makedirs(FAILED_ROOT, exist_ok=True)
    os.makedirs(PASS_ROOT, exist_ok=True)
    
    count_pass = 0
    count_fail = 0
    count_missing = 0
    total_files_copied = 0
    
    extensions = ['.jpg', '.json', '.xml']
    
    print(f"Starting to organize files from {SOURCE_ROOT}...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        # Extract metadata from CSV
        folder_name = str(row['Folder'])
        file_id = str(row['File_ID'])
        status = str(row['Status']).lower()
        
        # Cleanup folder name (in case it contains prefix like 'Quang_Part2_0314/')
        if '/' in folder_name:
            folder_name = folder_name.split('/')[-1]
            
        # Determine destination root
        if status == 'fail':
            dest_root = Path(FAILED_ROOT)
            count_fail += 1
        elif status == 'pass':
            dest_root = Path(PASS_ROOT)
            count_pass += 1
        else:
            continue
        
        # Paths
        src_dir = Path(SOURCE_ROOT) / folder_name
        dest_dir = dest_root / folder_name
        
        # Ensure subfolder exists in destination
        os.makedirs(dest_dir, exist_ok=True)
        
        item_found = False
        for ext in extensions:
            filename = f"{file_id}{ext}"
            src_path = src_dir / filename
            dest_path = dest_dir / filename
            
            # Copy file if it exists
            if src_path.exists():
                shutil.copy2(src_path, dest_path)
                total_files_copied += 1
                item_found = True
        
        if not item_found:
            count_missing += 1
            if count_missing <= 10:
                print(f"\nWarning: No files found for {folder_name}/{file_id} at {src_dir}")
            
    print(f"\nFinished organizing files.")
    print(f"Items processed: Pass={count_pass}, Fail={count_fail}")
    print(f"Total files copied (jpg+json+xml): {total_files_copied}")
    print(f"Items with no files found: {count_missing}")
    print(f"Output folders:")
    print(f" - Pass: {PASS_ROOT}")
    print(f" - Fail: {FAILED_ROOT}")

if __name__ == "__main__":
    organize_images()

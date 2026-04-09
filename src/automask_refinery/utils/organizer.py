import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional
from automask_refinery.utils.logger import log

class FileOrganizer:
    def __init__(self, source_root: str, pass_root: str, fail_root: str, csv_path: str):
        self.source_root = Path(source_root)
        self.pass_root = Path(pass_root)
        self.fail_root = Path(fail_root)
        self.csv_path = Path(csv_path)
        self.extensions = ['.jpg', '.json', '.xml', '.png', '.jpeg']

    def organize(self, move: bool = False):
        """
        Organizes files into pass/fail folders based on CSV status.
        :param move: If True, moves files instead of copying.
        """
        if not self.csv_path.exists():
            log.error(f"CSV file not found at {self.csv_path}")
            return

        log.info(f"Reading CSV from {self.csv_path}...")
        df = pd.read_csv(self.csv_path)
        
        self.pass_root.mkdir(parents=True, exist_ok=True)
        self.fail_root.mkdir(parents=True, exist_ok=True)
        
        counts = {"pass": 0, "fail": 0, "missing": 0}
        total_files = 0
        
        op_name = "Moving" if move else "Copying"
        log.info(f"{op_name} files from {self.source_root}...")
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=op_name):
            folder_name = str(row['Folder'])
            file_id = str(row['File_ID'])
            status = str(row['Status']).lower()
            
            # Decide destination
            if status == 'pass':
                dest_base = self.pass_root
                counts["pass"] += 1
            elif status == 'fail' or status == 'failed':
                dest_base = self.fail_root
                counts["fail"] += 1
            else:
                continue
            
            src_dir = self.source_root / folder_name
            dest_dir = dest_base / folder_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            found_any = False
            for ext in self.extensions:
                filename = f"{file_id}{ext}"
                # Also try uppercase extensions if not found
                for f_name in [filename, filename.upper()]:
                    src_file = src_dir / f_name
                    if src_file.exists():
                        dest_file = dest_dir / f_name
                        try:
                            if move:
                                shutil.move(str(src_file), str(dest_file))
                            else:
                                shutil.copy2(src_file, dest_file)
                            total_files += 1
                            found_any = True
                            break
                        except Exception as e:
                            log.error(f"Error {op_name.lower()} {src_file}: {e}")
            
            if not found_any:
                counts["missing"] += 1
        
        log.info(f"Organization complete.")
        log.info(f"Results: {counts}")
        log.info(f"Total files {op_name.lower()}: {total_files}")

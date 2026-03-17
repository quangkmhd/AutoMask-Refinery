import os
import csv
import shutil
from pathlib import Path

# Đường dẫn gốc
base_dir = Path("/home/quangnhvn34/dev/me/AutoMask-Refinery")
source_dir = base_dir / "logo_20260314/Quang_Part2_0314"
csv_path = base_dir / "review_details.csv"
pass_dir = base_dir / "pass"
failed_dir = base_dir / "failed"

# Các định dạng file cần chuyển
extensions = ['.jpg', '.jpeg', '.png', '.json', '.xml', '.JPG', '.PNG']

def organize_files():
    # Tạo thư mục pass và failed nếu chưa có
    pass_dir.mkdir(exist_ok=True)
    failed_dir.mkdir(exist_ok=True)

    if not csv_path.exists():
        print(f"Error: File {csv_path} does not exist.")
        return

    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        count_moved = 0
        count_missing = 0

        for row in reader:
            folder_name = row['Folder']
            file_id = row['File_ID']
            status = row['Status'].lower()
            
            # Quyết định thư mục đích dựa trên status
            dest_base = pass_dir if status == 'pass' else failed_dir
            
            # Thư mục nguồn cụ thể
            src_folder_path = source_dir / folder_name
            # Thư mục đích cụ thể (giữ nguyên cấu trúc folder để tránh trùng tên file_id)
            dest_folder_path = dest_base / folder_name
            
            if not src_folder_path.exists():
                # Thử tìm kiếm trực tiếp trong Quang_Part2 nếu folder_name không khớp hoàn toàn (một số trường hợp có thể thiếu prefix)
                # Nhưng dựa trên log thì có vẻ nó khớp.
                pass

            for ext in extensions:
                # Tên file có thể là file_id + ext
                filename = f"{file_id}{ext}"
                src_file = src_folder_path / filename
                
                if src_file.exists():
                    # Tạo thư mục đích nếu chưa có
                    dest_folder_path.mkdir(parents=True, exist_ok=True)
                    dest_file = dest_folder_path / filename
                    
                    # Di chuyển file
                    try:
                        shutil.move(str(src_file), str(dest_file))
                        count_moved += 1
                        # print(f"Moved: {src_file} -> {dest_file}")
                    except Exception as e:
                        print(f"Error moving {src_file}: {e}")
                else:
                    # Một số file có thể không đủ cả 3 loại (jpg, json, xml), đây là bình thường
                    pass

    print(f"Done! Moved {count_moved} files.")

if __name__ == "__main__":
    organize_files()

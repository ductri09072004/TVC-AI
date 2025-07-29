#!/usr/bin/env python3
"""
Script để tạo dataset nhỏ với 100 dòng đầu tiên từ dataset gốc
"""

import os
import shutil

def create_small_dataset(source_dir, target_dir, train_lines=5000, dev_lines=100, test_lines=100):
    """
    Tạo dataset nhỏ với số dòng chỉ định cho từng tập
    
    Args:
        source_dir: Thư mục chứa dataset gốc
        target_dir: Thư mục đích để lưu dataset nhỏ
        train_lines: Số dòng cho tập train (mặc định 1000)
        dev_lines: Số dòng cho tập dev (mặc định 100)
        test_lines: Số dòng cho tập test (mặc định 100)
    """
    
    # Tạo thư mục đích nếu chưa tồn tại
    os.makedirs(target_dir, exist_ok=True)
    
    # Các file cần copy
    files_to_copy = ['train.txt', 'dev.txt', 'test.txt', 'entities.txt', 'entity2text.txt', 'relations.txt', 'relation2text.txt']
    
    for filename in files_to_copy:
        source_file = os.path.join(source_dir, filename)
        target_file = os.path.join(target_dir, filename)
        
        if os.path.exists(source_file):
            if filename == 'train.txt':
                # Lấy số dòng cho train
                with open(source_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                selected_lines = lines[:train_lines]
                with open(target_file, 'w', encoding='utf-8') as f:
                    f.writelines(selected_lines)
                print(f"Created {target_file} with {len(selected_lines)} lines")
                
            elif filename == 'dev.txt':
                # Lấy số dòng cho dev
                with open(source_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                selected_lines = lines[:dev_lines]
                with open(target_file, 'w', encoding='utf-8') as f:
                    f.writelines(selected_lines)
                print(f"Created {target_file} with {len(selected_lines)} lines")
                
            elif filename == 'test.txt':
                # Lấy số dòng cho test
                with open(source_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                selected_lines = lines[:test_lines]
                with open(target_file, 'w', encoding='utf-8') as f:
                    f.writelines(selected_lines)
                print(f"Created {target_file} with {len(selected_lines)} lines")
            else:
                # Copy toàn bộ file cho các file metadata
                shutil.copy2(source_file, target_file)
                print(f"Copied {source_file} to {target_file}")
        else:
            print(f"Warning: {source_file} not found")

if __name__ == "__main__":
    # Tạo dataset nhỏ với tỷ lệ hợp lý
    create_small_dataset(
        source_dir="./data/VN13",
        target_dir="./data/VN13_small",
        train_lines=5000,  # 1000 dòng cho train
        dev_lines=100,     # 100 dòng cho dev
        test_lines=100     # 100 dòng cho test
    )
    
    print("\nDataset nhỏ đã được tạo tại ./data/VN13_small/")
    print("Tỷ lệ: Train=5000, Dev=100, Test=100")
    print("Bây giờ bạn có thể chạy BERT với dataset này:")
    print("python run_bert_triple_classifier.py --task_name kg --do_train --do_eval --do_predict --data_dir ./data/VN13_small --bert_model vinai/phobert-base --max_seq_length 20 --train_batch_size 32 --learning_rate 1e-5 --num_train_epochs 3.0 --output_dir ./output_VN13_small/ --gradient_accumulation_steps 1 --eval_batch_size 512") 
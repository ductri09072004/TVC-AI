import pandas as pd
import os
from pathlib import Path

def create_entities():
    # Đọc dữ liệu từ file gốc
    repo_root = Path(__file__).resolve().parents[1]  # .../BERT_NEW
    df = pd.read_csv(repo_root / 'DatasetTripletKGC.tsv', sep='\t', names=['head', 'relation', 'tail'])
    
    # Kết hợp tất cả các entity
    all_entities = set(df['head'].unique()) | set(df['tail'].unique())
    
    # Tạo thư mục nếu chưa tồn tại
    out_dir = repo_root / 'data' / 'KG_VNT'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Lưu vào file
    with open(out_dir / 'entities.txt', 'w', encoding='utf-8') as f:
        for entity in sorted(all_entities):
            f.write(f"{entity}\n")
    
    print(f"Đã tạo file entities.txt với {len(all_entities)} entities")

if __name__ == "__main__":
    create_entities() 
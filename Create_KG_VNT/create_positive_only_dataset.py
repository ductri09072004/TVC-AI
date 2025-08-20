import pandas as pd
from pathlib import Path

def create_positive_only_dataset():
    """Tạo dataset chỉ chứa positive examples từ file gốc"""
    
    # Đọc dữ liệu từ file gốc
    repo_root = Path(__file__).resolve().parents[1]  # .../BERT_NEW
    df = pd.read_csv(repo_root / 'DatasetTripletKGC.tsv', sep='\t', names=['head', 'relation', 'tail'])
    
    # Thêm label 1 cho tất cả (vì đây là positive triples từ file gốc)
    df['label'] = 1
    
    print(f"Original positive samples: {len(df)}")
    
    # Lưu vào file train.txt
    out_dir = repo_root / 'data' / 'KG_VNT'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Lưu file train.txt chỉ chứa positive examples
    df.to_csv(out_dir / 'train.txt', sep='\t', index=False, header=False)
    
    print(f"Đã tạo file train.txt với {len(df)} positive examples")
    print("Bây giờ bạn có thể chạy BERT model với Hard Negative Sampling tự động")

if __name__ == "__main__":
    create_positive_only_dataset()

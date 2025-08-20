import pandas as pd
from pathlib import Path

def create_relations():
    # Đọc dữ liệu từ file gốc
    repo_root = Path(__file__).resolve().parents[1]  # .../BERT_NEW
    df = pd.read_csv(repo_root / 'DatasetTripletKGC.tsv', sep='\t', names=['head', 'relation', 'tail'])
    
    # Lấy tất cả các relation
    all_relations = set(df['relation'].unique())
    
    # Tạo thư mục nếu chưa tồn tại
    out_dir = repo_root / 'data' / 'KG_VNT'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Lưu relations.txt
    with open(out_dir / 'relations.txt', 'w', encoding='utf-8') as f:
        for relation in sorted(all_relations):
            f.write(f"{relation}\n")
    
    # Lưu relation2text.txt
    with open(out_dir / 'relation2text.txt', 'w', encoding='utf-8') as f:
        for relation in sorted(all_relations):
            # Trong trường hợp này, chúng ta sẽ sử dụng chính relation làm text
            f.write(f"{relation}\t{relation}\n")
    
    print(f"Đã tạo file relations.txt và relation2text.txt với {len(all_relations)} relations")

if __name__ == "__main__":
    create_relations() 
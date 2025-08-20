import pandas as pd
from pathlib import Path

def create_entity2text():
    # Đọc dữ liệu từ file gốc
    repo_root = Path(__file__).resolve().parents[1]  # .../BERT_NEW
    df = pd.read_csv(repo_root / 'DatasetTripletKGC.tsv', sep='\t', names=['head', 'relation', 'tail'])
    
    # Kết hợp tất cả các entity
    all_entities = set(df['head'].unique()) | set(df['tail'].unique())
    
    # Tạo mapping từ entity sang text
    # Trong trường hợp này, chúng ta sẽ sử dụng chính entity làm text
    entity2text = {entity: entity for entity in all_entities}
    
    # Tạo thư mục nếu chưa tồn tại
    out_dir = repo_root / 'data' / 'KG_VNT'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Lưu vào file
    with open(out_dir / 'entity2text.txt', 'w', encoding='utf-8') as f:
        for entity, text in entity2text.items():
            f.write(f"{entity}\t{text}\n")
    
    print(f"Đã tạo file entity2text.txt với {len(entity2text)} entities")

if __name__ == "__main__":
    create_entity2text() 
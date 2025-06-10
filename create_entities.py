import pandas as pd
import os

def create_entities():
    # Đọc dữ liệu từ file gốc
    df = pd.read_csv('DatasetTripletKGC.tsv', sep='\t', names=['head', 'relation', 'tail'])
    
    # Kết hợp tất cả các entity
    all_entities = set(df['head'].unique()) | set(df['tail'].unique())
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs('data/VN13', exist_ok=True)
    
    # Lưu vào file
    with open('data/VN13/entities.txt', 'w', encoding='utf-8') as f:
        for entity in sorted(all_entities):
            f.write(f"{entity}\n")
    
    print(f"Đã tạo file entities.txt với {len(all_entities)} entities")

if __name__ == "__main__":
    create_entities() 
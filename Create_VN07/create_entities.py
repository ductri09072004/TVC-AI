import pandas as pd
import os

# Lệnh chạy
# python ./Create_VN07/create_entities.py
def create_entities():
    # Đọc dữ liệu từ file gốc
    df = pd.read_csv('Datasettriplets_VN07.tsv', sep='\t', names=['head', 'relation', 'tail'])
    
    # Kết hợp tất cả các entity
    all_entities = set(df['head'].unique()) | set(df['tail'].unique())
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs('data/VN07', exist_ok=True)
    
    # Lưu vào file
    with open('data/VN07/entities.txt', 'w', encoding='utf-8') as f:
        for entity in sorted(all_entities, key=lambda x: str(x)):
            f.write(f"{str(entity)}\n")
    
    print(f"Đã tạo file entities.txt với {len(all_entities)} entities")

if __name__ == "__main__":
    create_entities() 
import pandas as pd
import os

# Lệnh chạy
# python ./Create_VN07/create_entity2text.py
def create_entity2text():
    # Đọc dữ liệu từ file gốc
    df = pd.read_csv('Datasettriplets_VN07.tsv', sep='\t', names=['head', 'relation', 'tail'])
    
    # Kết hợp tất cả các entity
    all_entities = set(df['head'].unique()) | set(df['tail'].unique())
    
    # Tạo mapping từ entity sang text
    # Trong trường hợp này, chúng ta sẽ sử dụng chính entity làm text
    entity2text = {entity: entity for entity in all_entities}
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs('data/VN07', exist_ok=True)
    
    # Lưu vào file
    with open('data/VN07/entity2text.txt', 'w', encoding='utf-8') as f:
        for entity, text in entity2text.items():
            f.write(f"{entity}\t{text}\n")
    
    print(f"Đã tạo file entity2text.txt với {len(entity2text)} entities")

if __name__ == "__main__":
    create_entity2text() 
import pandas as pd
import os

# Lệnh chạy
# python ./Create_VN07/create_relations.py
def create_relations():
    # Đọc dữ liệu từ file gốc
    df = pd.read_csv('Datasettriplets_VN07.tsv', sep='\t', names=['head', 'relation', 'tail'])
    
    # Lấy tất cả các relation
    all_relations = set(df['relation'].unique())
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs('data/VN07', exist_ok=True)
    
    # Lưu relations.txt
    with open('data/VN07/relations.txt', 'w', encoding='utf-8') as f:
        for relation in sorted(all_relations):
            f.write(f"{relation}\n")
    
    # Lưu relation2text.txt
    with open('data/VN07/relation2text.txt', 'w', encoding='utf-8') as f:
        for relation in sorted(all_relations):
            # Trong trường hợp này, chúng ta sẽ sử dụng chính relation làm text
            f.write(f"{relation}\t{relation}\n")
    
    print(f"Đã tạo file relations.txt và relation2text.txt với {len(all_relations)} relations")

if __name__ == "__main__":
    create_relations() 
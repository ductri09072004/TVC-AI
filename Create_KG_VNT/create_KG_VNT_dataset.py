import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm

def create_hard_negative_samples(df):
    """Tạo Hard Negative samples từ positive samples với tỷ lệ 1:1"""
    # Lấy danh sách unique entities và relations
    all_entities = set(df['head'].unique()) | set(df['tail'].unique())
    all_relations = set(df['relation'].unique())
    
    # Tạo set tất cả positive triples để kiểm tra trùng lặp
    positive_triples_set = set()
    for _, row in df.iterrows():
        triple_str = f"{row['head']}\t{row['relation']}\t{row['tail']}"
        positive_triples_set.add(triple_str)
    
    negative_samples = []
    
    # Tạo Hard Negative samples với tỷ lệ 1:1
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating Hard Negative samples"):
        # Chọn ngẫu nhiên phương pháp tạo negative sample (head hoặc tail)
        method = random.choice(['head', 'tail'])
        
        if method == 'head':
            # Hard Negative: Thay đổi head entity
            attempts = 0
            max_attempts = 100  # Giới hạn số lần thử để tránh vòng lặp vô hạn
            
            while attempts < max_attempts:
                new_head = random.choice(list(all_entities - {row['head']}))
                new_triple_str = f"{new_head}\t{row['relation']}\t{row['tail']}"
                
                # Kiểm tra triple mới không trùng với positive triples
                if new_triple_str not in positive_triples_set:
                    negative_samples.append({
                        'head': new_head,
                        'relation': row['relation'],
                        'tail': row['tail'],
                        'label': 0
                    })
                    break
                attempts += 1
            
            # Nếu không tìm được, tạo random negative
            if attempts >= max_attempts:
                new_head = random.choice(list(all_entities - {row['head']}))
                negative_samples.append({
                    'head': new_head,
                    'relation': row['relation'],
                    'tail': row['tail'],
                    'label': 0
                })
                
        else:
            # Hard Negative: Thay đổi tail entity
            attempts = 0
            max_attempts = 100
            
            while attempts < max_attempts:
                new_tail = random.choice(list(all_entities - {row['tail']}))
                new_triple_str = f"{row['head']}\t{row['relation']}\t{new_tail}"
                
                # Kiểm tra triple mới không trùng với positive triples
                if new_triple_str not in positive_triples_set:
                    negative_samples.append({
                        'head': row['head'],
                        'relation': row['relation'],
                        'tail': new_tail,
                        'label': 0
                    })
                    break
                attempts += 1
            
            # Nếu không tìm được, tạo random negative
            if attempts >= max_attempts:
                new_tail = random.choice(list(all_entities - {row['tail']}))
                negative_samples.append({
                    'head': row['head'],
                    'relation': row['relation'],
                    'tail': new_tail,
                    'label': 0
                })
    
    return pd.DataFrame(negative_samples)

def augment_positive_samples(df):
    """Tăng cường positive samples bằng cách tạo các biến thể"""
    augmented_samples = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting positive samples"):
        # Thêm bản gốc
        augmented_samples.append({
            'head': row['head'],
            'relation': row['relation'],
            'tail': row['tail'],
            'label': 1
        })
        
        # Tạo biến thể bằng cách đảo vị trí head và tail (nếu có ý nghĩa)
        if random.random() < 0.3:  # 30% cơ hội tạo biến thể
            augmented_samples.append({
                'head': row['tail'],
                'relation': row['relation'],
                'tail': row['head'],
                'label': 1
            })
    
    return pd.DataFrame(augmented_samples, columns=['head', 'relation', 'tail', 'label'])

def main():
    # Đọc dữ liệu gốc
    print("Reading original data...")
    df = pd.read_csv('DatasetTripletKGC.tsv', sep='\t', names=['head', 'relation', 'tail'])
    
    # Thêm label 1 cho positive samples
    df['label'] = 1
    
    # Tăng cường positive samples
    print("Augmenting positive samples...")
    augmented_df = augment_positive_samples(df)
    
    # Tạo Hard Negative samples
    print("Creating Hard Negative samples...")
    negative_df = create_hard_negative_samples(augmented_df)
    
    # Kết hợp positive và negative samples
    combined_df = pd.concat([augmented_df, negative_df], ignore_index=True)
    
    # Xáo trộn dữ liệu
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Chia dữ liệu thành train, dev, test
    print("Splitting data into train, dev, test...")
    # Đầu tiên chia thành train và temp (dev + test)
    train_df, temp_df = train_test_split(combined_df, test_size=0.2, random_state=42)
    # Sau đó chia temp thành dev và test
    dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Lưu các file
    print("Saving files...")
    # Đảm bảo label là string "0" hoặc "1"
    train_df['label'] = train_df['label'].astype(str)
    dev_df['label'] = dev_df['label'].astype(str)
    test_df['label'] = test_df['label'].astype(str)
    
    train_df.to_csv('data/VN13/train.txt', sep='\t', index=False, header=False)
    dev_df.to_csv('data/VN13/dev.txt', sep='\t', index=False, header=False)
    test_df.to_csv('data/VN13/test.txt', sep='\t', index=False, header=False)
    
    print("Done!")
    print(f"Original positive samples: {len(df)}")
    print(f"Augmented positive samples: {len(augmented_df)}")
    print(f"Hard Negative samples: {len(negative_df)}")
    print(f"Train set size: {len(train_df)}")
    print(f"Dev set size: {len(dev_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # In thống kê về tỷ lệ các lớp
    print("\nClass distribution:")
    print("Train set:", train_df['label'].value_counts(normalize=True))
    print("Dev set:", dev_df['label'].value_counts(normalize=True))
    print("Test set:", test_df['label'].value_counts(normalize=True))

if __name__ == "__main__":
    main() 
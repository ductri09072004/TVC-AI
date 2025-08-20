import random

# Đường dẫn file
input_file = 'train.tsv'
entities_file = 'entities.txt'
output_file = 'train_10_with_neg.tsv'

# Đọc danh sách entity
with open(entities_file, encoding='utf-8') as f:
    entities = [line.strip() for line in f if line.strip()]

# Đọc các triple positive
with open(input_file, encoding='utf-8') as f:
    positives = [line.strip().split('\t') for line in f if line.strip()]

# Tạo set để kiểm tra nhanh triple positive
positive_set = set((subj, rel, obj) for subj, rel, obj in positives)

with open(output_file, 'w', encoding='utf-8') as out:
    for triple in positives:
        subj, rel, obj = triple
        # Ghi positive sample
        out.write(f"{subj}\t{rel}\t{obj}\t1\n")
        
        # Sinh negative sample bằng cách thay subject
        for _ in range(100):  # Thử tối đa 100 lần để tránh lặp vô hạn
            neg_subj = random.choice(entities)
            if neg_subj != subj and (neg_subj, rel, obj) not in positive_set:
                out.write(f"{neg_subj}\t{rel}\t{obj}\t0\n")
                break
        
        # Sinh negative sample bằng cách thay object
        for _ in range(100):
            neg_obj = random.choice(entities)
            if neg_obj != obj and (subj, rel, neg_obj) not in positive_set:
                out.write(f"{subj}\t{rel}\t{neg_obj}\t0\n")
                break

print(f"Đã sinh negative samples (loại bỏ âm giả) và lưu vào {output_file}") 
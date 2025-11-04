import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
import os
from tqdm import tqdm
import sys

# Lệnh chạy (tất cả-in-1)
# python ./Create_Data/create_dataset.py
def create_negative_samples(df_pos: pd.DataFrame) -> pd.DataFrame:
    """Tạo negative samples (hard negatives) từ positive samples.
    Nguyên tắc:
    - Chỉ thay head/tail bằng entity đã từng xuất hiện ở đúng vai trò với cùng relation (đảm bảo câu có vẻ hợp lý).
    - Chỉ thay relation bằng relation mà head đã từng có (hoặc tail đã từng có), nhưng (head, rel_new, tail) không tồn tại trong positives.
    - Luôn đảm bảo triple negative chưa tồn tại trong positives.
    """
    pos_set = set((h, r, t) for h, r, t in zip(df_pos['head'], df_pos['relation'], df_pos['tail']))

    # Chỉ số đồng xuất hiện để tạo negatives có ý nghĩa hơn
    rel_to_heads = {}
    rel_to_tails = {}
    head_rel_to_tails = {}
    tail_rel_to_heads = {}

    for h, r, t in zip(df_pos['head'], df_pos['relation'], df_pos['tail']):
        rel_to_heads.setdefault(r, set()).add(h)
        rel_to_tails.setdefault(r, set()).add(t)
        head_rel_to_tails.setdefault((h, r), set()).add(t)
        tail_rel_to_heads.setdefault((t, r), set()).add(h)

    all_relations = set(df_pos['relation'].unique())
    negative_samples = []

    for _, row in tqdm(df_pos.iterrows(), total=len(df_pos), desc="Creating hard negatives"):
        h = row['head']
        r = row['relation']
        t = row['tail']

        method = random.choice(['head', 'tail', 'relation'])
        created = False

        # Thử thay head: chọn head khác đã từng xuất hiện với relation r
        if not created and method in ('head',):
            candidates = list(rel_to_heads.get(r, set()) - {h})
            random.shuffle(candidates)
            for h2 in candidates[:50]:
                if (h2, r, t) not in pos_set:
                    negative_samples.append({'head': h2, 'relation': r, 'tail': t, 'label': 0})
                    created = True
                    break

        # Thử thay tail: chọn tail khác đã từng xuất hiện với relation r
        if not created and method in ('tail',):
            candidates = list(rel_to_tails.get(r, set()) - {t})
            random.shuffle(candidates)
            for t2 in candidates[:50]:
                if (h, r, t2) not in pos_set:
                    negative_samples.append({'head': h, 'relation': r, 'tail': t2, 'label': 0})
                    created = True
                    break

        # Thử thay relation: chọn relation khác mà head đã từng có (giữ domain hợp lý)
        if not created and method in ('relation',):
            # Quan hệ head đã từng có
            head_rels = set(rr for (hh, rr), tails in head_rel_to_tails.items() if hh == h)
            candidates = list((head_rels | all_relations) - {r})
            random.shuffle(candidates)
            for r2 in candidates[:50]:
                if (h, r2, t) not in pos_set:
                    negative_samples.append({'head': h, 'relation': r2, 'tail': t, 'label': 0})
                    created = True
                    break

        # Fallback: thử các phương án còn lại nếu phương án đã chọn không tạo được
        if not created:
            # head
            candidates = list(rel_to_heads.get(r, set()) - {h})
            random.shuffle(candidates)
            for h2 in candidates[:50]:
                if (h2, r, t) not in pos_set:
                    negative_samples.append({'head': h2, 'relation': r, 'tail': t, 'label': 0})
                    created = True
                    break
        if not created:
            # tail
            candidates = list(rel_to_tails.get(r, set()) - {t})
            random.shuffle(candidates)
            for t2 in candidates[:50]:
                if (h, r, t2) not in pos_set:
                    negative_samples.append({'head': h, 'relation': r, 'tail': t2, 'label': 0})
                    created = True
                    break
        if not created:
            # relation
            head_rels = set(rr for (hh, rr), tails in head_rel_to_tails.items() if hh == h)
            candidates = list((head_rels | all_relations) - {r})
            random.shuffle(candidates)
            for r2 in candidates[:50]:
                if (h, r2, t) not in pos_set:
                    negative_samples.append({'head': h, 'relation': r2, 'tail': t, 'label': 0})
                    created = True
                    break

        # Nếu vẫn không tạo được, bỏ qua để giữ chất lượng negatives

    return pd.DataFrame(negative_samples)


def write_entities(df_svo: pd.DataFrame, out_dir: str = 'dataset') -> None:
    os.makedirs(out_dir, exist_ok=True)
    # Normalize helper
    def norm(x: str) -> str:
        if pd.isna(x):
            return ''
        s = str(x).strip().strip('"').strip("'")
        s = s.replace('(', '').replace(')', '')
        return ' '.join(s.split())

    s_series = df_svo['s'].astype(str).map(norm)
    o_series = df_svo['o'].astype(str).map(norm)

    subs = set(s_series.unique()) - {''}
    objs = set(o_series.unique()) - {''}
    all_entities = subs | objs
    # Backward-compat combined
    with open(os.path.join(out_dir, 'entities.txt'), 'w', encoding='utf-8') as f:
        for entity in sorted(all_entities, key=lambda x: str(x)):
            f.write(f"{str(entity)}\n")
    # Separate roles
    # Sort by frequency (desc), then alphabetically
    s_counts = s_series.value_counts()
    o_counts = o_series.value_counts()
    with open(os.path.join(out_dir, 'subjects.txt'), 'w', encoding='utf-8') as fs:
        for entity in sorted(subs, key=lambda x: (-int(s_counts.get(x, 0)), x)):
            fs.write(f"{str(entity)}\n")
    with open(os.path.join(out_dir, 'objects.txt'), 'w', encoding='utf-8') as fo:
        for entity in sorted(objs, key=lambda x: (-int(o_counts.get(x, 0)), x)):
            fo.write(f"{str(entity)}\n")


def write_entity2text(df_svo: pd.DataFrame, out_dir: str = 'dataset') -> None:
    os.makedirs(out_dir, exist_ok=True)
    # Normalize helper
    def norm(x: str) -> str:
        if pd.isna(x):
            return ''
        s = str(x).strip().strip('"').strip("'")
        s = s.replace('(', '').replace(')', '')
        return ' '.join(s.split())

    s_series = df_svo['s'].astype(str).map(norm)
    o_series = df_svo['o'].astype(str).map(norm)

    subs = set(s_series.unique()) - {''}
    objs = set(o_series.unique()) - {''}
    all_entities = subs | objs
    # Backward-compat combined
    entity2text = {entity: entity for entity in all_entities}
    with open(os.path.join(out_dir, 'entity2text.txt'), 'w', encoding='utf-8') as f:
        for entity, text in entity2text.items():
            f.write(f"{entity}\t{text}\n")
    # Separate roles
    s_counts = s_series.value_counts()
    o_counts = o_series.value_counts()
    with open(os.path.join(out_dir, 'subject2text.txt'), 'w', encoding='utf-8') as fs:
        for entity in sorted(subs, key=lambda x: (-int(s_counts.get(x, 0)), x)):
            fs.write(f"{entity}\t{entity}\n")
    with open(os.path.join(out_dir, 'object2text.txt'), 'w', encoding='utf-8') as fo:
        for entity in sorted(objs, key=lambda x: (-int(o_counts.get(x, 0)), x)):
            fo.write(f"{entity}\t{entity}\n")


def write_relations(df_svo: pd.DataFrame, out_dir: str = 'dataset') -> None:
    os.makedirs(out_dir, exist_ok=True)
    all_relations = set(df_svo['v'].dropna().astype(str).unique())
    all_relations = {r for r in all_relations if r.strip() and r != 'nan'}
    # relations.txt
    with open(os.path.join(out_dir, 'relations.txt'), 'w', encoding='utf-8') as f:
        for relation in sorted(all_relations):
            f.write(f"{relation}\n")
    # relation2text.txt
    with open(os.path.join(out_dir, 'relation2text.txt'), 'w', encoding='utf-8') as f:
        for relation in sorted(all_relations):
            f.write(f"{relation}\t{relation}\n")

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
        # Không đảo head/tail mặc định để tránh tạo câu vô nghĩa
    
    return pd.DataFrame(augmented_samples, columns=['head', 'relation', 'tail', 'label'])

def main():
    # Đọc dữ liệu: ưu tiên dataset_svo.csv; nếu chưa có, chuyển từ dataset.csv (caption,label)
    svo_path = 'data/dataset_svo.csv'
    raw_path = 'data/dataset.csv'
    if os.path.isfile(svo_path):
        print("Reading original data from dataset_svo.csv...")
        df_svo = pd.read_csv(svo_path, encoding='utf-8')
    else:
        print("dataset_svo.csv not found. Converting from dataset.csv (caption,label) to SVO...")
        if not os.path.isfile(raw_path):
            raise FileNotFoundError("Neither data/dataset_svo.csv nor data/dataset.csv found.")
        # Import extract_svo
        ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
        sys.path.insert(0, ROOT)
        try:
            from convert_to_svo import extract_svo  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Cannot import extract_svo for conversion: {e}")
        raw_df = pd.read_csv(raw_path, encoding='utf-8', names=None)
        # Detect header for dataset.csv
        if set(map(str.lower, raw_df.columns)) >= { 'caption', 'label' }:
            cap_col = 'caption'
            lab_col = 'label'
        else:
            # Assume no header: two columns caption,label
            if raw_df.shape[1] < 2:
                raise ValueError('dataset.csv must have at least 2 columns: caption,label')
            raw_df.columns = ['caption', 'label'] + [f'c{i}' for i in range(2, raw_df.shape[1])]
            cap_col = 'caption'
            lab_col = 'label'
        captions = raw_df[cap_col].astype(str).fillna('')
        labels = pd.to_numeric(raw_df[lab_col], errors='coerce').fillna(0).astype(int)
        s_list, v_list, o_list = [], [], []
        for txt in tqdm(captions, desc='Converting to SVO'):
            s, v, o = extract_svo(str(txt))
            s_list.append(s)
            v_list.append(v)
            o_list.append(o)
        df_svo = pd.DataFrame({ 's': s_list, 'v': v_list, 'o': o_list, 'label': labels })
        # Optional: save the converted file for reuse
        try:
            os.makedirs('data', exist_ok=True)
            df_svo.to_csv(svo_path, index=False, encoding='utf-8')
            print(f"Saved converted SVO to {svo_path}")
        except Exception:
            pass

    # Bước 1: Tạo entities, entity2text, relations
    print("Writing entities/entities2text/relations to ./dataset ...")
    write_entities(df_svo, out_dir='dataset')
    write_entity2text(df_svo, out_dir='dataset')
    write_relations(df_svo, out_dir='dataset')
    
    # Chuyển đổi từ format s,v,o,label sang head,relation,tail,label (GIỮ NGUYÊN NHÃN có sẵn)
    # Convert label sang số an toàn (handle "0"/"1")
    df = pd.DataFrame({
        'head': df_svo['s'],
        'relation': df_svo['v'],
        'tail': df_svo['o'],
        'label': pd.to_numeric(df_svo['label'], errors='coerce').fillna(0).astype(int)
    })

    # Remove parentheses from fields
    # and clean literal 'nan' that may slip in when casting
    for _col in ['head', 'relation', 'tail']:
        df[_col] = (
            df[_col]
            .astype(str)
            .str.replace(r'[()]', '', regex=True)
            .str.strip()
        )
        # Turn string 'nan' (any case) into empty string
        df[_col] = df[_col].apply(lambda x: '' if str(x).strip().lower() == 'nan' else x)

    # Try to infer missing relation from head+tail using extract_svo
    try:
        from convert_to_svo import extract_svo  # type: ignore
        def _infer_rel(row):
            rel = str(row['relation']).strip()
            if rel:
                return rel
            s = str(row['head']).strip()
            t = str(row['tail']).strip()
            merged = f"{s} {t}".strip()
            s2, v2, o2 = extract_svo(merged)
            v2 = (v2 or '').strip()
            return v2
        df['relation'] = df.apply(_infer_rel, axis=1)
    except Exception:
        # If extractor not available, keep as-is
        pass

    # Drop any rows that still miss relation after inference
    df = df[df['relation'].astype(str).str.strip() != '']

    # Loại bỏ các dòng có giá trị rỗng trong head, relation, hoặc tail
    df = df.dropna(subset=['head', 'relation', 'tail'])
    df = df[df['head'].astype(str).str.strip() != '']
    df = df[df['relation'].astype(str).str.strip() != '']
    df = df[df['tail'].astype(str).str.strip() != '']
    
    # KHÔNG tạo phủ định tổng hợp; dùng trực tiếp nhãn sẵn có trong dataset_svo.csv
    total = len(df)
    num_pos = int((df['label'] == 1).sum())
    num_neg = int((df['label'] == 0).sum())
    print(f"Total samples: {total}")
    print(f"Positive (label=1): {num_pos}")
    print(f"Negative (label=0): {num_neg}")
    
    combined_df = df.copy()
    
    # Xáo trộn dữ liệu
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Chia dữ liệu thành train, dev, test (giữ tỉ lệ lớp)
    print("Splitting data into train, dev, test...")
    # Đầu tiên chia thành train và temp (dev + test)
    train_df, temp_df = train_test_split(combined_df, test_size=0.2, random_state=42, stratify=combined_df['label'] if combined_df['label'].nunique()>1 else None)
    # Sau đó chia temp thành dev và test
    dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'] if temp_df['label'].nunique()>1 else None)
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs('dataset', exist_ok=True)
    
    # Lưu các file
    print("Saving files...")
    # Đảm bảo label là string "0" hoặc "1"
    train_df['label'] = train_df['label'].astype(str)
    dev_df['label'] = dev_df['label'].astype(str)
    test_df['label'] = test_df['label'].astype(str)
    
    train_df.to_csv('dataset/train.txt', sep='\t', index=False, header=False)
    dev_df.to_csv('dataset/dev.txt', sep='\t', index=False, header=False)
    test_df.to_csv('dataset/test.txt', sep='\t', index=False, header=False)
    
    print("Done!")
    print(f"Positives used: {num_pos}")
    print(f"Negatives used: {num_neg}")
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
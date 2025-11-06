"""
Script chia dataset thành train/dev/test (80/10/10)
và tạo các file cần thiết cho training BERT
"""

import pandas as pd
import os
import numpy as np
from collections import defaultdict
from typing import Optional, Tuple, Dict, Any
import io
import argparse
import json

# Cấu hình mặc định (có thể override qua CLI)
INPUT_FILE = "data/dataset_real.csv"
OUTPUT_DIR = "dataset/dataset_svo"
TRAIN_RATIO = 0.8
DEV_RATIO = 0.1
TEST_RATIO = 0.1
# Tên cột dùng để group (đảm bảo cùng video/campaign không rơi vào split khác nhau).
# Nếu None, script sẽ cố gắng tự đoán: ['video_id','video','file','filename','source_video','uid']
GROUP_COL: Optional[str] = None
RANDOM_STATE = 42
WRITE_JSON_SUMMARY = True

def ensure_output_dir():
    """Tạo thư mục output nếu chưa có"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Created/verified output directory: {OUTPUT_DIR}")

def load_dataset():
    """Đọc dataset với xử lý lỗi CSV (curly quotes, dấu phẩy trong trường)."""
    print(f"Reading dataset from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE, encoding='utf-8')
        print(f"Loaded {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Primary CSV parse failed: {e}. Retrying with normalization and python engine...")
        # Đọc toàn bộ file, chuẩn hóa curly quotes → \" để pandas nhận diện quotechar
        with open(INPUT_FILE, 'r', encoding='utf-8', errors='replace') as f:
            raw = f.read()
        normalized = (
            raw.replace('\u201c', '"')  # “
               .replace('\u201d', '"')  # ”
               .replace('“', '"')
               .replace('”', '"')
        )
        try:
            df = pd.read_csv(
                io.StringIO(normalized),
                encoding='utf-8',
                engine='python',
                sep=',',
                quotechar='"',
                on_bad_lines='warn',
            )
            print(f"Loaded {len(df)} rows (python engine)")
            print(f"Columns: {df.columns.tolist()}")
            return df
        except Exception as e2:
            # Phương án cuối: bỏ qua dòng lỗi để tiếp tục
            print(f"Python engine also failed: {e2}. Falling back to skip bad lines...")
            df = pd.read_csv(
                INPUT_FILE,
                encoding='utf-8',
                engine='python',
                sep=',',
                quotechar='"',
                on_bad_lines='skip',
            )
            print(f"Loaded {len(df)} rows (skipped bad lines)")
            print(f"Columns: {df.columns.tolist()}")
            return df

def _detect_group_column(df: pd.DataFrame) -> Optional[str]:
    if GROUP_COL and GROUP_COL in df.columns:
        return GROUP_COL
    candidates = [
        "video_id", "video", "file", "filename", "source_video", "uid",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _two_stage_stratified_group_split(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trả về index cho train/dev/test theo tỉ lệ yêu cầu với (stratified)+(group).
    Chiến lược: tách 80% train vs 20% temp, sau đó tách temp → dev/test 50/50.
    """
    n = len(X)
    idx_all = np.arange(n)
    # Import chiến lược split
    sgk = None
    try:
        from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
        sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    except Exception:
        sgk = None

    # Helper: lấy fold gần nhất với tỉ lệ mong muốn
    def split_with_ratio(indices, labels, groups_arr, ratio):
        # Chọn một fold làm validation với kích thước gần ratio
        if groups_arr is not None and sgk is not None:
            best = None
            for train_idx, val_idx in sgk.split(indices, labels, groups_arr):
                frac = len(val_idx) / len(indices)
                score = abs(frac - ratio)
                if best is None or score < best[0]:
                    best = (score, train_idx, val_idx)
            assert best is not None
            return indices[best[1]], indices[best[2]]
        else:
            # Fallback: StratifiedKFold (không group)
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            best = None
            for tr, va in skf.split(indices, labels):
                frac = len(va) / len(indices)
                score = abs(frac - ratio)
                if best is None or score < best[0]:
                    best = (score, tr, va)
            assert best is not None
            return indices[best[1]], indices[best[2]]

    labels = y.astype(int)
    group_arr = groups if groups is not None else None
    # Bước 1: Train vs Temp
    train_idx, temp_idx = split_with_ratio(idx_all, labels, group_arr, 1.0 - train_ratio)
    # Bước 2: Temp → Dev/Test (50/50 của temp nếu dev=test)
    temp_labels = labels[temp_idx]
    temp_groups = group_arr[temp_idx] if group_arr is not None else None
    # Tránh chia 0
    temp_total = len(temp_idx)
    desired_dev_ratio_in_temp = dev_ratio / (dev_ratio + test_ratio) if (dev_ratio + test_ratio) > 0 else 0.5
    dev_sub_idx, test_sub_idx = split_with_ratio(np.arange(temp_total), temp_labels, temp_groups, 1.0 - desired_dev_ratio_in_temp)
    dev_idx = temp_idx[dev_sub_idx]
    test_idx = temp_idx[test_sub_idx]
    return train_idx, dev_idx, test_idx


def split_dataset(df):
    """Chia dataset thành train/dev/test (stratified + optional grouping theo video)"""
    # Xóa dòng có giá trị NaN
    df = df.dropna(subset=['s', 'v', 'o', 'label'])
    
    # Đảm bảo label là 0 hoặc 1
    # Chuyển đổi label một cách an toàn
    def normalize_label(val):
        try:
            if pd.isna(val):
                return None
            val_str = str(val).strip().lower()
            # Nếu là header hoặc text không hợp lệ, bỏ qua
            if val_str in ['label', '']:
                return None
            val_int = int(float(val_str))
            return val_int if val_int in [0, 1] else None
        except:
            return None
    
    df = df.copy()  # Tránh SettingWithCopyWarning
    df['label_clean'] = df['label'].apply(normalize_label)
    df = df.dropna(subset=['label_clean'])
    df['label'] = df['label_clean'].astype(int)
    df = df.drop(columns=['label_clean'])
    
    print(f"After cleaning: {len(df)} rows")
    
    # Tìm cột group (nếu có)
    group_col = _detect_group_column(df)
    if group_col:
        print(f"Using group column: {group_col}")
        groups = df[group_col].astype(str).fillna("").values
    else:
        print("No group column found. Will use stratified split without grouping.")
        groups = None

    # Tạo chỉ số split theo chiến lược đề xuất
    y = df['label'].astype(int).values
    train_idx, dev_idx, test_idx = _two_stage_stratified_group_split(
        df, y, groups, TRAIN_RATIO, DEV_RATIO, TEST_RATIO, random_state=RANDOM_STATE
    )
    # Lấy các dataframe theo index
    train_df = df.iloc[train_idx].copy()
    dev_df = df.iloc[dev_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    print(f"\nSplit results:")
    n_total = len(df)
    print(f"  Train: {len(train_df)} rows ({len(train_df)/n_total*100:.1f}%)")
    print(f"  Dev:   {len(dev_df)} rows ({len(dev_df)/n_total*100:.1f}%)")
    print(f"  Test:  {len(test_df)} rows ({len(test_df)/n_total*100:.1f}%)")
    
    return train_df, dev_df, test_df

def create_triple_files(train_df, dev_df, test_df):
    """Tạo các file train.txt, dev.txt, test.txt với format: subject\tverb\tobject\tlabel"""
    print("\nCreating triple files...")
    
    def write_triple_file(df, filename):
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                s = str(row['s']).strip()
                v = str(row['v']).strip()
                o = str(row['o']).strip()
                label = str(int(row['label']))
                f.write(f"{s}\t{v}\t{o}\t{label}\n")
        print(f"  Created {filepath} ({len(df)} rows)")
    
    write_triple_file(train_df, "train.txt")
    write_triple_file(dev_df, "dev.txt")
    write_triple_file(test_df, "test.txt")

    # Ghi manifest id để tái lập (index gốc sau shuffle ở đây là index của df sau cleaning)
    for name, part in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        manifest_path = os.path.join(OUTPUT_DIR, f"{name}_manifest.csv")
        cols = [c for c in ["s","v","o","label"] if c in part.columns]
        extra_cols = []
        group_col = _detect_group_column(part)
        if group_col and group_col not in cols:
            extra_cols.append(group_col)
        part.reset_index(drop=True)[cols + extra_cols].to_csv(manifest_path, index=False, encoding="utf-8")
        print(f"  Created {manifest_path} ({len(part)} rows)")

def _compute_split_stats(df: pd.DataFrame, name: str, group_col: Optional[str]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "name": name,
        "num_rows": int(len(df)),
        "label_0": int((df["label"] == 0).sum()) if "label" in df.columns else None,
        "label_1": int((df["label"] == 1).sum()) if "label" in df.columns else None,
    }
    if stats["num_rows"] and "label" in df.columns:
        stats["pct_label_0"] = round(stats["label_0"] / stats["num_rows"] * 100.0, 2) if stats["label_0"] is not None else None
        stats["pct_label_1"] = round(stats["label_1"] / stats["num_rows"] * 100.0, 2) if stats["label_1"] is not None else None
    if group_col and group_col in df.columns:
        stats["num_groups"] = int(df[group_col].nunique())
    return stats

def _check_group_leakage(train_df: pd.DataFrame, dev_df: pd.DataFrame, test_df: pd.DataFrame, group_col: Optional[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"group_col": group_col, "leakages": {}}
    if not group_col or group_col not in train_df.columns:
        return result
    g_train = set(train_df[group_col].astype(str))
    g_dev = set(dev_df[group_col].astype(str))
    g_test = set(test_df[group_col].astype(str))
    leak_td = sorted(list(g_train & g_dev))
    leak_tt = sorted(list(g_train & g_test))
    leak_dt = sorted(list(g_dev & g_test))
    result["leakages"]["train_dev"] = leak_td[:20]
    result["leakages"]["train_test"] = leak_tt[:20]
    result["leakages"]["dev_test"] = leak_dt[:20]
    result["has_leakage"] = any([leak_td, leak_tt, leak_dt])
    return result

def create_entity_mapping_files(train_df, dev_df, test_df):
    """Tạo các file mapping cho subject, object, relation"""
    print("\nCreating entity mapping files...")
    
    # Lấy tất cả subjects từ cả 3 sets
    all_subjects = set()
    all_objects = set()
    all_relations = set()
    
    for df in [train_df, dev_df, test_df]:
        all_subjects.update(df['s'].astype(str).str.strip())
        all_objects.update(df['o'].astype(str).str.strip())
        all_relations.update(df['v'].astype(str).str.strip())
    
    # Tạo mapping cho subjects
    print(f"  Found {len(all_subjects)} unique subjects")
    subject2text_path = os.path.join(OUTPUT_DIR, "subject2text.txt")
    with open(subject2text_path, 'w', encoding='utf-8') as f:
        for idx, subject in enumerate(sorted(all_subjects), 1):
            if subject:
                f.write(f"{subject}\t{subject}\n")
    print(f"  Created {subject2text_path}")
    
    # Tạo mapping cho objects
    print(f"  Found {len(all_objects)} unique objects")
    object2text_path = os.path.join(OUTPUT_DIR, "object2text.txt")
    with open(object2text_path, 'w', encoding='utf-8') as f:
        for idx, obj in enumerate(sorted(all_objects), 1):
            if obj:
                f.write(f"{obj}\t{obj}\n")
    print(f"  Created {object2text_path}")
    
    # Tạo mapping cho relations (verbs)
    print(f"  Found {len(all_relations)} unique relations")
    relation2text_path = os.path.join(OUTPUT_DIR, "relation2text.txt")
    with open(relation2text_path, 'w', encoding='utf-8') as f:
        for idx, rel in enumerate(sorted(all_relations), 1):
            if rel:
                f.write(f"{rel}\t{rel}\n")
    print(f"  Created {relation2text_path}")
    
    # Tạo file entities.txt (tổng hợp subjects và objects)
    print(f"  Creating entities file...")
    all_entities = sorted(all_subjects.union(all_objects))
    entities_path = os.path.join(OUTPUT_DIR, "entities.txt")
    with open(entities_path, 'w', encoding='utf-8') as f:
        for entity in all_entities:
            if entity:
                f.write(f"{entity}\n")
    print(f"  Created {entities_path} ({len(all_entities)} entities)")
    
    # Tạo file entity2text.txt (mapping entity -> text)
    entity2text_path = os.path.join(OUTPUT_DIR, "entity2text.txt")
    with open(entity2text_path, 'w', encoding='utf-8') as f:
        for entity in all_entities:
            if entity:
                f.write(f"{entity}\t{entity}\n")
    print(f"  Created {entity2text_path}")
    
    # Tạo file subjects.txt
    subjects_path = os.path.join(OUTPUT_DIR, "subjects.txt")
    with open(subjects_path, 'w', encoding='utf-8') as f:
        for subject in sorted(all_subjects):
            if subject:
                f.write(f"{subject}\n")
    print(f"  Created {subjects_path}")
    
    # Tạo file objects.txt
    objects_path = os.path.join(OUTPUT_DIR, "objects.txt")
    with open(objects_path, 'w', encoding='utf-8') as f:
        for obj in sorted(all_objects):
            if obj:
                f.write(f"{obj}\n")
    print(f"  Created {objects_path}")
    
    # Tạo file relations.txt
    relations_path = os.path.join(OUTPUT_DIR, "relations.txt")
    with open(relations_path, 'w', encoding='utf-8') as f:
        for rel in sorted(all_relations):
            if rel:
                f.write(f"{rel}\n")
    print(f"  Created {relations_path}")

def create_statistics(train_df, dev_df, test_df):
    """Tạo file thống kê"""
    print("\nCreating statistics...")
    
    stats = []
    stats.append("=== Dataset Statistics ===\n")
    stats.append(f"Total rows: {len(train_df) + len(dev_df) + len(test_df)}\n")
    stats.append(f"Train: {len(train_df)} rows\n")
    stats.append(f"Dev: {len(dev_df)} rows\n")
    stats.append(f"Test: {len(test_df)} rows\n\n")
    
    for name, df in [("Train", train_df), ("Dev", dev_df), ("Test", test_df)]:
        stats.append(f"=== {name} Set ===\n")
        stats.append(f"Total: {len(df)} rows\n")
        stats.append(f"Label 0: {(df['label'] == 0).sum()} rows ({(df['label'] == 0).sum()/len(df)*100:.1f}%)\n")
        stats.append(f"Label 1: {(df['label'] == 1).sum()} rows ({(df['label'] == 1).sum()/len(df)*100:.1f}%)\n\n")
    
    stats_path = os.path.join(OUTPUT_DIR, "dataset_stats.txt")
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.writelines(stats)
    print(f"  Created {stats_path}")
    # JSON summary (tùy chọn)
    if WRITE_JSON_SUMMARY:
        group_col = _detect_group_column(pd.concat([train_df, dev_df, test_df], axis=0, ignore_index=True))
        json_stats = {
            "splits": [
                _compute_split_stats(train_df, "train", group_col),
                _compute_split_stats(dev_df, "dev", group_col),
                _compute_split_stats(test_df, "test", group_col),
            ]
        }
        leakage = _check_group_leakage(train_df, dev_df, test_df, group_col)
        json_stats["group_leakage"] = leakage
        json_path = os.path.join(OUTPUT_DIR, "dataset_stats.json")
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(json_stats, jf, ensure_ascii=False, indent=2)
        print(f"  Created {json_path}")

def main():
    """Hàm chính"""
    global INPUT_FILE, OUTPUT_DIR, TRAIN_RATIO, DEV_RATIO, TEST_RATIO, GROUP_COL, RANDOM_STATE, WRITE_JSON_SUMMARY

    parser = argparse.ArgumentParser(description="Split dataset into train/dev/test with stratified group strategy")
    parser.add_argument("--input", default=INPUT_FILE, help="Đường dẫn file CSV đầu vào")
    parser.add_argument("--out_dir", default=OUTPUT_DIR, help="Thư mục output (dataset_svo)")
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--dev_ratio", type=float, default=DEV_RATIO)
    parser.add_argument("--test_ratio", type=float, default=TEST_RATIO)
    parser.add_argument("--group_col", type=str, default=GROUP_COL, help="Tên cột group (tùy chọn)")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--no_json_summary", action="store_true", help="Không ghi file dataset_stats.json")
    args = parser.parse_args()

    # Apply overrides
    INPUT_FILE = args.input
    OUTPUT_DIR = args.out_dir
    TRAIN_RATIO = args.train_ratio
    DEV_RATIO = args.dev_ratio
    TEST_RATIO = args.test_ratio
    GROUP_COL = args.group_col
    RANDOM_STATE = args.seed
    WRITE_JSON_SUMMARY = not args.no_json_summary

    # Validate ratios
    total_ratio = TRAIN_RATIO + DEV_RATIO + TEST_RATIO
    if not (0.999 <= total_ratio <= 1.001):
        raise ValueError(f"Tổng tỷ lệ phải = 1.0, hiện tại = {total_ratio}")
    if min(TRAIN_RATIO, DEV_RATIO, TEST_RATIO) < 0:
        raise ValueError("Các tỷ lệ không được âm")

    print("=" * 60)
    print("Dataset Splitter for BERT Training")
    print("=" * 60)
    print(f"Config: input={INPUT_FILE}, out_dir={OUTPUT_DIR}, ratios=({TRAIN_RATIO},{DEV_RATIO},{TEST_RATIO}), seed={RANDOM_STATE}, group_col={GROUP_COL}")
    
    # Tạo thư mục output
    ensure_output_dir()
    
    # Đọc dataset
    df = load_dataset()
    
    # Chia dataset
    train_df, dev_df, test_df = split_dataset(df)
    
    # Tạo các file triple (train.txt, dev.txt, test.txt)
    create_triple_files(train_df, dev_df, test_df)
    
    # Tạo các file mapping (subject2text.txt, object2text.txt, etc.)
    create_entity_mapping_files(train_df, dev_df, test_df)
    
    # Tạo file thống kê
    create_statistics(train_df, dev_df, test_df)
    
    print("\n" + "=" * 60)
    print("Completed successfully!")
    print(f"All files saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)

if __name__ == "__main__":
    main()

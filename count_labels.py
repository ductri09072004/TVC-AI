"""
Script để thống kê số lượng nhãn 0 và 1 trong dataset CSV
"""
import pandas as pd
import sys
import os
from pathlib import Path
from collections import Counter

# Fix encoding for Windows console
if sys.platform == 'win32':
    try:
        import codecs
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except:
        # Fallback: set console to UTF-8
        os.system('chcp 65001 >nul 2>&1')

def count_labels(input_file, show_details=True):
    """
    Thống kê số lượng nhãn 0 và 1 trong dataset CSV
    
    Parameters:
    -----------
    input_file : str
        Đường dẫn đến file CSV đầu vào
    show_details : bool, default True
        Hiển thị thông tin chi tiết về phân bố nhãn
    """
    print(f"Đang đọc file: {input_file}")
    
    # Đọc file CSV
    try:
        df = pd.read_csv(input_file)
        print(f"✓ Đã đọc thành công {len(df)} dòng dữ liệu\n")
    except Exception as e:
        print(f"✗ Lỗi khi đọc file: {e}")
        return
    
    # Kiểm tra cột label có tồn tại không
    if 'label' not in df.columns:
        print("✗ Không tìm thấy cột 'label' trong file CSV")
        print(f"Các cột có sẵn: {list(df.columns)}")
        return
    
    # Chuyển đổi label về dạng số nguyên một cách an toàn
    # Xử lý các trường hợp: float ("0.0", "1.0"), int ("0", "1"), hoặc giá trị không hợp lệ
    def convert_label(value):
        try:
            # Thử chuyển đổi sang float trước (để xử lý "0.0", "1.0")
            float_val = float(value)
            int_val = int(float_val)
            # Chỉ chấp nhận 0 hoặc 1
            if int_val in [0, 1]:
                return int_val
            else:
                return None
        except (ValueError, TypeError):
            return None
    
    # Áp dụng chuyển đổi
    df['label'] = df['label'].apply(convert_label)
    
    # Loại bỏ các dòng có label không hợp lệ
    invalid_count = df['label'].isna().sum()
    if invalid_count > 0:
        print(f"⚠ Cảnh báo: Tìm thấy {invalid_count} dòng có nhãn không hợp lệ (sẽ bị bỏ qua)")
        df = df.dropna(subset=['label'])
    
    # Chuyển đổi sang int
    df['label'] = df['label'].astype(int)
    
    # Đếm số lượng từng nhãn
    label_counts = df['label'].value_counts().sort_index()
    
    # Tổng số dòng
    total = len(df)
    
    # Hiển thị kết quả
    print("="*60)
    print("THỐNG KÊ NHÃN TRONG DATASET")
    print("="*60)
    print(f"\nTổng số dòng dữ liệu: {total:,}")
    print(f"\nSố lượng từng nhãn:")
    print("-" * 60)
    
    for label in sorted(label_counts.index):
        count = label_counts[label]
        percentage = (count / total) * 100
        print(f"  Nhãn {label}: {count:,} dòng ({percentage:.2f}%)")
    
    # Kiểm tra nhãn không hợp lệ (sau khi đã xử lý)
    valid_labels = [0, 1]
    invalid_labels = df[~df['label'].isin(valid_labels)]['label'].unique()
    if len(invalid_labels) > 0:
        print(f"\n⚠ Cảnh báo: Vẫn còn nhãn không hợp lệ: {list(invalid_labels)}")
        invalid_count = len(df[~df['label'].isin(valid_labels)])
        print(f"  Số dòng có nhãn không hợp lệ: {invalid_count}")
    
    # Tính tỷ lệ cân bằng
    if 0 in label_counts.index and 1 in label_counts.index:
        count_0 = label_counts[0]
        count_1 = label_counts[1]
        ratio = min(count_0, count_1) / max(count_0, count_1)
        
        print(f"\nTỷ lệ cân bằng (balance ratio): {ratio:.4f}")
        if ratio < 0.5:
            print("  ⚠ Dataset bị mất cân bằng nghiêm trọng (ratio < 0.5)")
        elif ratio < 0.7:
            print("  ⚠ Dataset hơi mất cân bằng (0.5 <= ratio < 0.7)")
        else:
            print("  ✓ Dataset khá cân bằng (ratio >= 0.7)")
    
    # Hiển thị thông tin chi tiết nếu được yêu cầu
    if show_details:
        print("\n" + "="*60)
        print("CHI TIẾT PHÂN BỐ")
        print("="*60)
        
        # Thống kê theo từng cột nếu có
        if 's' in df.columns:
            print(f"\nSố lượng giá trị unique trong cột 's': {df['s'].nunique()}")
        if 'v' in df.columns:
            print(f"Số lượng giá trị unique trong cột 'v': {df['v'].nunique()}")
        if 'o' in df.columns:
            print(f"Số lượng giá trị unique trong cột 'o': {df['o'].nunique()}")
        
        # Top 10 giá trị phổ biến nhất trong cột 'o' cho mỗi nhãn
        if 'o' in df.columns:
            print("\nTop 10 giá trị 'o' phổ biến nhất cho nhãn 0:")
            top_o_label_0 = df[df['label'] == 0]['o'].value_counts().head(10)
            for idx, (value, count) in enumerate(top_o_label_0.items(), 1):
                print(f"  {idx}. {value[:60]}... ({count} lần)")
            
            print("\nTop 10 giá trị 'o' phổ biến nhất cho nhãn 1:")
            top_o_label_1 = df[df['label'] == 1]['o'].value_counts().head(10)
            for idx, (value, count) in enumerate(top_o_label_1.items(), 1):
                print(f"  {idx}. {value[:60]}... ({count} lần)")
    
    # Tạo báo cáo tóm tắt
    print("\n" + "="*60)
    print("BÁO CÁO TÓM TẮT")
    print("="*60)
    print(f"File: {input_file}")
    print(f"Tổng số dòng: {total:,}")
    if 0 in label_counts.index:
        print(f"Nhãn 0: {label_counts[0]:,} ({(label_counts[0]/total)*100:.2f}%)")
    if 1 in label_counts.index:
        print(f"Nhãn 1: {label_counts[1]:,} ({(label_counts[1]/total)*100:.2f}%)")
    
    return {
        'total': total,
        'label_0': label_counts.get(0, 0),
        'label_1': label_counts.get(1, 0),
        'ratio': ratio if (0 in label_counts.index and 1 in label_counts.index) else None
    }


if __name__ == "__main__":
    # Đường dẫn mặc định
    default_input = "data/dataset_real.csv"
    
    # Kiểm tra tham số dòng lệnh
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = default_input
    
    # Kiểm tra file có tồn tại không
    if not Path(input_file).exists():
        print(f"✗ Lỗi: Không tìm thấy file {input_file}")
        print(f"\nCách sử dụng:")
        print(f"  python count_labels.py [input_file]")
        print(f"\nVí dụ:")
        print(f"  python count_labels.py data/dataset_real.csv")
        sys.exit(1)
    
    # Tùy chọn: hiển thị chi tiết hay không
    show_details = True
    if len(sys.argv) > 2:
        if sys.argv[2].lower() in ['false', '0', 'no', 'simple']:
            show_details = False
    
    # Chạy hàm thống kê
    result = count_labels(input_file, show_details)


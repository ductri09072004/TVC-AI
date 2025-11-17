"""
Script để kiểm tra và xóa dữ liệu trùng trong dataset_real.csv
"""
import pandas as pd
import sys
from pathlib import Path

def remove_duplicates(input_file, output_file=None, check_all_columns=True, keep='first'):
    """
    Xóa các dòng trùng lặp trong dataset CSV
    
    Parameters:
    -----------
    input_file : str
        Đường dẫn đến file CSV đầu vào
    output_file : str, optional
        Đường dẫn đến file CSV đầu ra. Nếu None, sẽ ghi đè file đầu vào
    check_all_columns : bool, default True
        Nếu True: kiểm tra trùng dựa trên tất cả các cột
        Nếu False: chỉ kiểm tra trùng dựa trên cột s, v, o (bỏ qua label)
    keep : str, default 'first'
        'first': giữ lại dòng đầu tiên khi gặp trùng
        'last': giữ lại dòng cuối cùng khi gặp trùng
    """
    print(f"Đang đọc file: {input_file}")
    
    # Đọc file CSV
    try:
        df = pd.read_csv(input_file)
        print(f"Tổng số dòng ban đầu: {len(df)}")
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return
    
    # Hiển thị thông tin về dataset
    print(f"\nCác cột trong dataset: {list(df.columns)}")
    print(f"Số dòng trước khi xóa trùng: {len(df)}")
    
    # Xác định cột để kiểm tra trùng
    if check_all_columns:
        subset = None  # Kiểm tra tất cả các cột
        print("Kiểm tra trùng dựa trên TẤT CẢ các cột")
    else:
        subset = ['s', 'v', 'o']  # Chỉ kiểm tra s, v, o
        print("Kiểm tra trùng dựa trên các cột: s, v, o (bỏ qua label)")
    
    # Đếm số dòng trùng
    duplicates = df.duplicated(subset=subset, keep=False)
    num_duplicates = duplicates.sum()
    
    if num_duplicates > 0:
        print(f"\nTìm thấy {num_duplicates} dòng bị trùng")
        
        # Hiển thị một số ví dụ về dòng trùng
        print("\nMột số ví dụ về dòng trùng:")
        duplicate_rows = df[duplicates]
        for idx, row in duplicate_rows.head(10).iterrows():
            print(f"  Dòng {idx + 2}: {row.to_dict()}")
        
        if len(duplicate_rows) > 10:
            print(f"  ... và {len(duplicate_rows) - 10} dòng trùng khác")
    else:
        print("\nKhông tìm thấy dòng trùng nào!")
        return
    
    # Xóa trùng, giữ lại dòng đầu tiên (hoặc cuối cùng)
    df_cleaned = df.drop_duplicates(subset=subset, keep=keep)
    
    print(f"\nSố dòng sau khi xóa trùng: {len(df_cleaned)}")
    print(f"Đã xóa {len(df) - len(df_cleaned)} dòng trùng")
    
    # Xác định file đầu ra
    if output_file is None:
        output_file = input_file
        print(f"\nĐang ghi đè file: {output_file}")
    else:
        print(f"\nĐang lưu vào file mới: {output_file}")
    
    # Lưu file đã làm sạch
    try:
        df_cleaned.to_csv(output_file, index=False)
        print(f"✓ Đã lưu thành công!")
    except Exception as e:
        print(f"Lỗi khi lưu file: {e}")
        return
    
    # Tạo báo cáo
    print("\n" + "="*50)
    print("BÁO CÁO")
    print("="*50)
    print(f"File đầu vào: {input_file}")
    print(f"Số dòng ban đầu: {len(df)}")
    print(f"Số dòng sau khi xóa trùng: {len(df_cleaned)}")
    print(f"Số dòng đã xóa: {len(df) - len(df_cleaned)}")
    print(f"File đầu ra: {output_file}")


if __name__ == "__main__":
    # Đường dẫn mặc định
    default_input = "data/dataset_negative.csv"
    
    # Kiểm tra tham số dòng lệnh
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = default_input
    
    # Kiểm tra file có tồn tại không
    if not Path(input_file).exists():
        print(f"Lỗi: Không tìm thấy file {input_file}")
        sys.exit(1)
    
    # Tùy chọn: file đầu ra (nếu không có thì ghi đè file đầu vào)
    output_file = None
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    # Tùy chọn: kiểm tra trùng dựa trên tất cả cột hay chỉ s, v, o
    check_all = True
    if len(sys.argv) > 3:
        if sys.argv[3].lower() in ['false', '0', 'no', 'svo']:
            check_all = False
    
    # Tùy chọn: giữ dòng nào khi trùng
    keep_option = 'first'
    if len(sys.argv) > 4:
        if sys.argv[4].lower() in ['last', 'cuoi']:
            keep_option = 'last'
    
    print("="*50)
    print("SCRIPT XÓA DỮ LIỆU TRÙNG")
    print("="*50)
    print(f"\nCách sử dụng:")
    print(f"  python remove_duplicates.py [input_file] [output_file] [check_all] [keep]")
    print(f"\nTham số:")
    print(f"  input_file:   File CSV đầu vào (mặc định: {default_input})")
    print(f"  output_file:  File CSV đầu ra (mặc định: ghi đè file đầu vào)")
    print(f"  check_all:    'true' để kiểm tra tất cả cột, 'false' để chỉ kiểm tra s,v,o (mặc định: true)")
    print(f"  keep:         'first' để giữ dòng đầu, 'last' để giữ dòng cuối (mặc định: first)")
    print("\n" + "="*50 + "\n")
    
    # Chạy hàm xóa trùng
    remove_duplicates(input_file, output_file, check_all, keep_option)


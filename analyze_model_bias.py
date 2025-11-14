"""
Script để phân tích bias của model và tìm các trường hợp phân loại sai
"""
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

# Fix encoding for Windows console
if sys.platform == 'win32':
    try:
        import codecs
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except:
        os.system('chcp 65001 >nul 2>&1')

def parse_probabilities(line):
    """Parse probability line from test_results.tsv"""
    line = line.strip()
    # Remove brackets and split
    line = line.replace('[', '').replace(']', '')
    probs = [float(x.strip()) for x in line.split()]
    return probs

def analyze_model_bias(test_results_file, test_labels_file, threshold=0.5, output_dir=None):
    """
    Phân tích bias của model
    
    Parameters:
    -----------
    test_results_file : str
        Đường dẫn đến file test_results.tsv chứa xác suất dự đoán
    test_labels_file : str
        Đường dẫn đến file test.csv chứa labels thực tế
    threshold : float, default 0.5
        Ngưỡng để phân loại (nếu prob_class_1 > threshold thì dự đoán là 1)
    output_dir : str, optional
        Thư mục để lưu kết quả phân tích
    """
    print("="*70)
    print("PHÂN TÍCH BIAS CỦA MODEL")
    print("="*70)
    
    # Đọc test results (xác suất)
    print(f"\nĐang đọc file dự đoán: {test_results_file}")
    predictions_probs = []
    try:
        with open(test_results_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    probs = parse_probabilities(line)
                    predictions_probs.append(probs)
        print(f"✓ Đã đọc {len(predictions_probs)} dự đoán")
    except Exception as e:
        print(f"✗ Lỗi khi đọc file dự đoán: {e}")
        return
    
    # Đọc test labels
    print(f"\nĐang đọc file labels: {test_labels_file}")
    try:
        df_test = pd.read_csv(test_labels_file)
        print(f"✓ Đã đọc {len(df_test)} labels")
        
        # Lấy labels thực tế
        if 'label' in df_test.columns:
            true_labels = df_test['label'].astype(int).values
        elif 'caption' in df_test.columns and len(df_test.columns) == 2:
            # Giả sử cột thứ 2 là label
            true_labels = df_test.iloc[:, 1].astype(int).values
        else:
            print("✗ Không tìm thấy cột label trong file test")
            return
    except Exception as e:
        print(f"✗ Lỗi khi đọc file labels: {e}")
        return
    
    # Kiểm tra số lượng khớp
    if len(predictions_probs) != len(true_labels):
        print(f"\n⚠ Cảnh báo: Số lượng dự đoán ({len(predictions_probs)}) khác số lượng labels ({len(true_labels)})")
        min_len = min(len(predictions_probs), len(true_labels))
        predictions_probs = predictions_probs[:min_len]
        true_labels = true_labels[:min_len]
        print(f"  Chỉ phân tích {min_len} mẫu đầu tiên")
    
    # Chuyển đổi xác suất thành dự đoán
    predictions_probs = np.array(predictions_probs)
    # predictions_probs có dạng [prob_class_0, prob_class_1]
    prob_class_1 = predictions_probs[:, 1]
    predictions = (prob_class_1 > threshold).astype(int)
    
    # Phân tích phân bố
    print("\n" + "="*70)
    print("PHÂN BỐ DỮ LIỆU")
    print("="*70)
    
    true_label_counts = pd.Series(true_labels).value_counts().sort_index()
    pred_label_counts = pd.Series(predictions).value_counts().sort_index()
    
    print(f"\nPhân bố labels THỰC TẾ:")
    for label in sorted(true_label_counts.index):
        count = true_label_counts[label]
        pct = (count / len(true_labels)) * 100
        print(f"  Label {label}: {count:,} ({pct:.2f}%)")
    
    print(f"\nPhân bố labels DỰ ĐOÁN (threshold={threshold}):")
    for label in sorted(pred_label_counts.index):
        count = pred_label_counts[label]
        pct = (count / len(predictions)) * 100
        print(f"  Label {label}: {count:,} ({pct:.2f}%)")
    
    # Tính toán metrics
    print("\n" + "="*70)
    print("METRICS CHI TIẾT")
    print("="*70)
    
    cm = confusion_matrix(true_labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, average=None, zero_division=0
    )
    
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                 0       1")
    print(f"Actual  0    {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"        1    {cm[1,0]:6d}  {cm[1,1]:6d}")
    
    print("\nMetrics cho từng class:")
    print(f"  Class 0 (Không vi phạm):")
    print(f"    Precision: {precision[0]:.4f}")
    print(f"    Recall:    {recall[0]:.4f}")
    print(f"    F1-score:  {f1[0]:.4f}")
    print(f"    Support:   {support[0]:,}")
    
    print(f"\n  Class 1 (Vi phạm):")
    print(f"    Precision: {precision[1]:.4f}")
    print(f"    Recall:    {recall[1]:.4f}")
    print(f"    F1-score:  {f1[1]:.4f}")
    print(f"    Support:   {support[1]:,}")
    
    # Phân tích các trường hợp sai
    print("\n" + "="*70)
    print("PHÂN TÍCH CÁC TRƯỜNG HỢP SAI")
    print("="*70)
    
    # False Positive: Dự đoán 1 nhưng thực tế là 0 (quan trọng nhất!)
    false_positives = (predictions == 1) & (true_labels == 0)
    fp_count = false_positives.sum()
    fp_indices = np.where(false_positives)[0]
    
    # False Negative: Dự đoán 0 nhưng thực tế là 1
    false_negatives = (predictions == 0) & (true_labels == 1)
    fn_count = false_negatives.sum()
    fn_indices = np.where(false_negatives)[0]
    
    print(f"\nFalse Positives (FP): {fp_count:,} trường hợp")
    print(f"  → Dự đoán VI PHẠM nhưng thực tế KHÔNG VI PHẠM")
    print(f"  → Đây là vấn đề nghiêm trọng nhất!")
    
    print(f"\nFalse Negatives (FN): {fn_count:,} trường hợp")
    print(f"  → Dự đoán KHÔNG VI PHẠM nhưng thực tế VI PHẠM")
    
    # Hiển thị một số ví dụ False Positive
    if fp_count > 0 and 'caption' in df_test.columns:
        print("\n" + "-"*70)
        print("VÍ DỤ FALSE POSITIVE (Dự đoán sai - Vi phạm nhưng thực tế không vi phạm):")
        print("-"*70)
        num_examples = min(10, fp_count)
        for i, idx in enumerate(fp_indices[:num_examples], 1):
            caption = df_test.iloc[idx]['caption'] if idx < len(df_test) else f"Index {idx}"
            prob_0 = predictions_probs[idx][0]
            prob_1 = predictions_probs[idx][1]
            print(f"\n{i}. {caption[:80]}...")
            print(f"   Xác suất: Class 0 = {prob_0:.4f}, Class 1 = {prob_1:.4f}")
            print(f"   Dự đoán: {predictions[idx]} (threshold={threshold})")
            print(f"   Thực tế: {true_labels[idx]}")
    
    # Phân tích xác suất
    print("\n" + "="*70)
    print("PHÂN TÍCH XÁC SUẤT")
    print("="*70)
    
    fp_probs = prob_class_1[false_positives] if fp_count > 0 else np.array([])
    fn_probs = prob_class_1[false_negatives] if fn_count > 0 else np.array([])
    tp_probs = prob_class_1[(predictions == 1) & (true_labels == 1)] if (predictions == 1).sum() > 0 else np.array([])
    tn_probs = prob_class_1[(predictions == 0) & (true_labels == 0)] if (predictions == 0).sum() > 0 else np.array([])
    
    if len(fp_probs) > 0:
        print(f"\nXác suất Class 1 cho False Positives:")
        print(f"  Mean: {fp_probs.mean():.4f}")
        print(f"  Median: {np.median(fp_probs):.4f}")
        print(f"  Min: {fp_probs.min():.4f}")
        print(f"  Max: {fp_probs.max():.4f}")
        print(f"  Std: {fp_probs.std():.4f}")
    
    # Đề xuất threshold mới
    print("\n" + "="*70)
    print("ĐỀ XUẤT GIẢI PHÁP")
    print("="*70)
    
    # Thử các threshold khác nhau
    print("\nThử nghiệm với các threshold khác nhau:")
    print("-"*70)
    print(f"{'Threshold':<12} {'FP':<10} {'FN':<10} {'Precision_0':<15} {'Recall_0':<15} {'F1_0':<10}")
    print("-"*70)
    
    best_threshold = threshold
    best_fp = fp_count
    thresholds_to_try = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for thresh in thresholds_to_try:
        preds = (prob_class_1 > thresh).astype(int)
        fp = ((preds == 1) & (true_labels == 0)).sum()
        fn = ((preds == 0) & (true_labels == 1)).sum()
        prec, rec, f1_score, _ = precision_recall_fscore_support(
            true_labels, preds, average=None, zero_division=0
        )
        print(f"{thresh:<12.1f} {fp:<10} {fn:<10} {prec[0]:<15.4f} {rec[0]:<15.4f} {f1_score[0]:<10.4f}")
        
        # Tìm threshold tốt nhất (giảm FP nhưng không tăng FN quá nhiều)
        if fp < best_fp and fn < len(true_labels) * 0.1:  # FN không quá 10%
            best_threshold = thresh
            best_fp = fp
    
    print(f"\n→ Đề xuất threshold: {best_threshold:.2f} (giảm FP từ {fp_count} xuống ~{best_fp})")
    
    # Đề xuất class weights
    print("\nĐề xuất điều chỉnh Class Weights trong training:")
    class_0_count = (true_labels == 0).sum()
    class_1_count = (true_labels == 1).sum()
    total = len(true_labels)
    
    # Tính weight để cân bằng
    weight_0 = total / (2 * class_0_count)
    weight_1 = total / (2 * class_1_count)
    
    print(f"  Class 0 weight: {weight_0:.4f}")
    print(f"  Class 1 weight: {weight_1:.4f}")
    print(f"  (Tỷ lệ hiện tại: Class 0 = {class_0_count/total:.2%}, Class 1 = {class_1_count/total:.2%})")
    
    # Lưu kết quả nếu có output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Lưu False Positives
        if fp_count > 0 and 'caption' in df_test.columns:
            fp_df = df_test.iloc[fp_indices].copy()
            fp_df['predicted_prob_class_0'] = predictions_probs[fp_indices, 0]
            fp_df['predicted_prob_class_1'] = predictions_probs[fp_indices, 1]
            fp_df['predicted_label'] = predictions[fp_indices]
            fp_file = os.path.join(output_dir, 'false_positives.csv')
            fp_df.to_csv(fp_file, index=False, encoding='utf-8')
            print(f"\n✓ Đã lưu {fp_count} False Positives vào: {fp_file}")
        
        # Lưu báo cáo
        report_file = os.path.join(output_dir, 'bias_analysis_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("BÁO CÁO PHÂN TÍCH BIAS MODEL\n")
            f.write("="*70 + "\n\n")
            f.write(f"False Positives: {fp_count}\n")
            f.write(f"False Negatives: {fn_count}\n")
            f.write(f"Đề xuất threshold: {best_threshold:.2f}\n")
            f.write(f"Class 0 weight: {weight_0:.4f}\n")
            f.write(f"Class 1 weight: {weight_1:.4f}\n")
        print(f"✓ Đã lưu báo cáo vào: {report_file}")


if __name__ == "__main__":
    # Đường dẫn mặc định
    default_test_results = "output_moderation/test_results.tsv"
    default_test_labels = "dataset/dataset_mota/test.csv"
    
    # Kiểm tra tham số
    if len(sys.argv) > 1:
        test_results_file = sys.argv[1]
    else:
        test_results_file = default_test_results
    
    if len(sys.argv) > 2:
        test_labels_file = sys.argv[2]
    else:
        test_labels_file = default_test_labels
    
    # Threshold
    threshold = 0.5
    if len(sys.argv) > 3:
        try:
            threshold = float(sys.argv[3])
        except:
            pass
    
    # Output directory
    output_dir = None
    if len(sys.argv) > 4:
        output_dir = sys.argv[4]
    
    # Kiểm tra files
    if not Path(test_results_file).exists():
        print(f"✗ Không tìm thấy file: {test_results_file}")
        sys.exit(1)
    
    if not Path(test_labels_file).exists():
        print(f"✗ Không tìm thấy file: {test_labels_file}")
        sys.exit(1)
    
    # Chạy phân tích
    analyze_model_bias(test_results_file, test_labels_file, threshold, output_dir)


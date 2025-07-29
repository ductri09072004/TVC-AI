#!/usr/bin/env python3
"""
Script so sánh công bằng giữa BERT và GPT
"""

import os
import json

def read_eval_results(file_path):
    """Đọc file eval_results.txt"""
    results = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                    results[key] = value
    return results

def compare_results():
    """So sánh kết quả BERT và GPT"""
    
    print("=" * 60)
    print("SO SÁNH CÔNG BẰNG: BERT vs GPT")
    print("=" * 60)
    
    # Đọc kết quả BERT trên dev
    bert_dev_results = read_eval_results("./output_VN13_100/eval_results.txt")
    
    # Đọc kết quả GPT trên test
    gpt_test_results = read_eval_results("./output_gpt_VN13_100/test_results.txt")
    
    print("\n📊 KẾT QUẢ HIỆN TẠI:")
    print("-" * 40)
    
    if bert_dev_results:
        print(f"BERT (Dev set):")
        print(f"  Accuracy: {bert_dev_results.get('eval_accuracy', 'N/A')}")
        print(f"  Precision: {bert_dev_results.get('eval_precision', 'N/A')}")
        print(f"  Recall: {bert_dev_results.get('eval_recall', 'N/A')}")
        print(f"  F1-Score: {bert_dev_results.get('eval_f1_score', 'N/A')}")
        print(f"  Loss: {bert_dev_results.get('eval_loss', 'N/A')}")
    
    if gpt_test_results:
        print(f"\nGPT (Test set):")
        print(f"  Accuracy: {gpt_test_results.get('test_eval_accuracy', 'N/A')}")
        print(f"  Precision: {gpt_test_results.get('test_precision', 'N/A')}")
        print(f"  Recall: {gpt_test_results.get('test_recall', 'N/A')}")
        print(f"  F1-Score: {gpt_test_results.get('test_f1_score', 'N/A')}")
    
    print("\n⚠️  LƯU Ý:")
    print("- BERT đánh giá trên Dev set")
    print("- GPT đánh giá trên Test set")
    print("- Không thể so sánh trực tiếp do khác tập dữ liệu")
    
    print("\n💡 ĐỂ SO SÁNH CÔNG BẰNG:")
    print("1. Chạy GPT trên Dev set:")
    print("   python run_gpt_triple_classifier.py --data_dir ./data/VN13_100 --output_dir ./output_gpt_VN13_100_dev --api_key YOUR_API_KEY --do_eval")
    print("\n2. Chạy BERT trên Test set:")
    print("   python run_bert_triple_classifier.py --task_name kg --do_predict --data_dir ./data/VN13_100 --bert_model vinai/phobert-base --max_seq_length 20 --train_batch_size 32 --learning_rate 1e-5 --num_train_epochs 3.0 --output_dir ./output_VN13_100_test/ --gradient_accumulation_steps 1 --eval_batch_size 512")

if __name__ == "__main__":
    compare_results() 
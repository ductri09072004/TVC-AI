#!/usr/bin/env python3
"""
Script để so sánh kết quả giữa BERT chuẩn, PhoBERT và GPT versions
"""

import os
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any

def read_eval_results(file_path: str) -> Dict[str, Any]:
    """Đọc file eval_results.txt và parse kết quả"""
    results = {}
    
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} does not exist")
        return results
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Try to convert to float if possible
                try:
                    value = float(value)
                except ValueError:
                    pass
                
                results[key] = value
    
    return results

def create_comparison_charts(bert_std_values: list, phobert_values: list, gpt_values: list, metric_names: list, output_file: str) -> None:
    """Tạo biểu đồ so sánh BERT chuẩn vs PhoBERT vs GPT"""
    
    # Set up the figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Bar chart
    x = np.arange(len(metric_names))
    width = 0.25
    
    bars1 = ax1.bar(x - width, bert_std_values, width, label='BERT Standard', color='#1f77b4', alpha=0.8)
    bars2 = ax1.bar(x, phobert_values, width, label='PhoBERT', color='#ff7f0e', alpha=0.8)
    bars3 = ax1.bar(x + width, gpt_values, width, label='GPT', color='#2ca02c', alpha=0.8)
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('BERT Standard vs PhoBERT vs GPT Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    for bar in bars3:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    bert_std_values_radar = bert_std_values + bert_std_values[:1]
    phobert_values_radar = phobert_values + phobert_values[:1]
    gpt_values_radar = gpt_values + gpt_values[:1]
    
    ax2.plot(angles, bert_std_values_radar, 'o-', linewidth=2, label='BERT Standard', color='#1f77b4')
    ax2.fill(angles, bert_std_values_radar, alpha=0.25, color='#1f77b4')
    ax2.plot(angles, phobert_values_radar, 'o-', linewidth=2, label='PhoBERT', color='#ff7f0e')
    ax2.fill(angles, phobert_values_radar, alpha=0.25, color='#ff7f0e')
    ax2.plot(angles, gpt_values_radar, 'o-', linewidth=2, label='GPT', color='#2ca02c')
    ax2.fill(angles, gpt_values_radar, alpha=0.25, color='#2ca02c')
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metric_names)
    ax2.set_ylim(0, 1)
    ax2.set_title('Performance Radar Chart')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0))
    ax2.grid(True)
    
    # Add summary statistics
    bert_std_avg = np.mean([v for v in bert_std_values if v > 0])
    phobert_avg = np.mean([v for v in phobert_values if v > 0])
    gpt_avg = np.mean([v for v in gpt_values if v > 0])
    
    fig.suptitle(f'BERT Standard vs PhoBERT vs GPT Triple Classification Comparison\n'
                f'Average Performance: BERT_Std={bert_std_avg:.3f}, PhoBERT={phobert_avg:.3f}, GPT={gpt_avg:.3f}', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save as PDF
    pdf_filename = output_file.replace('.json', '_comparison.pdf')
    plt.savefig(pdf_filename, format='pdf', dpi=300, bbox_inches='tight')
    print(f"\n📊 Biểu đồ đã được lưu: {pdf_filename}")
    
    # Save as PNG for preview
    png_filename = output_file.replace('.json', '_comparison.png')
    plt.savefig(png_filename, format='png', dpi=300, bbox_inches='tight')
    print(f"📈 Biểu đồ PNG: {png_filename}")
    
    plt.show()

def compare_results(bert_std_results: Dict[str, Any], phobert_results: Dict[str, Any], gpt_results: Dict[str, Any], output_file: str = "comparison_results.json") -> None:
    """So sánh và hiển thị kết quả giữa BERT chuẩn, PhoBERT và GPT"""
    
    print("=" * 80)
    print("COMPARISON: BERT Standard vs PhoBERT vs GPT Triple Classification Results")
    print("=" * 80)
    
    # Common metrics to compare (4 chỉ số chính)
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    print(f"{'Metric':<20} {'BERT_Std':<15} {'PhoBERT':<15} {'GPT':<15} {'Best':<10}")
    print("-" * 80)
    
    bert_std_values = []
    phobert_values = []
    gpt_values = []
    
    for metric in metrics:
        # Tìm giá trị cho BERT Standard (có thể có prefix "eval_")
        bert_std_val = bert_std_results.get(f'eval_{metric}', bert_std_results.get(metric, 0))
        
        # Tìm giá trị cho PhoBERT (có thể có prefix "eval_")
        phobert_val = phobert_results.get(f'eval_{metric}', phobert_results.get(metric, 0))
        
        # Tìm giá trị cho GPT (có thể có prefix "eval_" hoặc không)
        gpt_val = gpt_results.get(f'eval_{metric}', gpt_results.get(metric, 0))
        
        bert_std_values.append(bert_std_val)
        phobert_values.append(phobert_val)
        gpt_values.append(gpt_val)
        
        # Tìm model tốt nhất cho metric này
        values = [bert_std_val, phobert_val, gpt_val]
        models = ['BERT_Std', 'PhoBERT', 'GPT']
        best_idx = np.argmax(values)
        best_model = models[best_idx] if values[best_idx] > 0 else 'N/A'
        
        print(f"{metric:<20} {str(bert_std_val):<15} {str(phobert_val):<15} {str(gpt_val):<15} {best_model:<10}")
    
    print("-" * 80)
    
    # Additional analysis
    print("\nAdditional Analysis:")
    print("-" * 40)
    
    # Tìm accuracy cho cả 3 models
    bert_std_acc = bert_std_results.get('eval_accuracy', bert_std_results.get('accuracy', 0))
    phobert_acc = phobert_results.get('eval_accuracy', phobert_results.get('accuracy', 0))
    gpt_acc = gpt_results.get('eval_accuracy', gpt_results.get('accuracy', 0))
    
    accuracies = [
        ('BERT Standard', bert_std_acc),
        ('PhoBERT', phobert_acc),
        ('GPT', gpt_acc)
    ]
    
    # Sắp xếp theo accuracy giảm dần
    accuracies.sort(key=lambda x: x[1], reverse=True)
    
    print("Ranking by Accuracy:")
    for i, (model, acc) in enumerate(accuracies, 1):
        if acc > 0:
            print(f"{i}. {model}: {acc:.4f}")
    
    # So sánh BERT Standard vs PhoBERT
    if bert_std_acc > 0 and phobert_acc > 0:
        diff = bert_std_acc - phobert_acc
        if diff > 0:
            print(f"\nBERT Standard performs better than PhoBERT by {diff:.4f} ({diff/phobert_acc*100:.2f}%)")
        elif diff < 0:
            print(f"\nPhoBERT performs better than BERT Standard by {abs(diff):.4f} ({abs(diff)/bert_std_acc*100:.2f}%)")
        else:
            print(f"\nBERT Standard and PhoBERT perform equally")
    
    # So sánh BERT Standard vs GPT
    if bert_std_acc > 0 and gpt_acc > 0:
        diff = bert_std_acc - gpt_acc
        if diff > 0:
            print(f"BERT Standard performs better than GPT by {diff:.4f} ({diff/gpt_acc*100:.2f}%)")
        elif diff < 0:
            print(f"GPT performs better than BERT Standard by {abs(diff):.4f} ({abs(diff)/bert_std_acc*100:.2f}%)")
        else:
            print(f"BERT Standard and GPT perform equally")
    
    # Create visualization
    create_comparison_charts(bert_std_values, phobert_values, gpt_values, metric_names, output_file)
    
    # Check for additional metrics
    bert_std_keys = set(bert_std_results.keys())
    phobert_keys = set(phobert_results.keys())
    gpt_keys = set(gpt_results.keys())
    
    all_keys = bert_std_keys | phobert_keys | gpt_keys
    common_keys = bert_std_keys & phobert_keys & gpt_keys
    
    print(f"\nMetrics available:")
    print(f"- BERT Standard: {len(bert_std_keys)} metrics")
    print(f"- PhoBERT: {len(phobert_keys)} metrics")
    print(f"- GPT: {len(gpt_keys)} metrics")
    print(f"- Common to all: {len(common_keys)} metrics")

def main():
    parser = argparse.ArgumentParser(description="Compare BERT Standard, PhoBERT and GPT results")
    parser.add_argument("--bert_std_output_dir", required=True, 
                        help="BERT Standard output directory containing eval_results.txt")
    parser.add_argument("--phobert_output_dir", required=True,
                        help="PhoBERT output directory containing eval_results.txt")
    parser.add_argument("--gpt_output_dir", required=True,
                        help="GPT output directory containing eval_results.txt")
    parser.add_argument("--output_file", default="comparison_results.json",
                        help="Output file to save comparison results")
    
    args = parser.parse_args()
    
    # Read results
    bert_std_results = read_eval_results(os.path.join(args.bert_std_output_dir, "eval_results.txt"))
    phobert_results = read_eval_results(os.path.join(args.phobert_output_dir, "eval_results.txt"))
    gpt_results = read_eval_results(os.path.join(args.gpt_output_dir, "eval_results.txt"))
    
    if not bert_std_results:
        print(f"Error: No BERT Standard results found in {args.bert_std_output_dir}")
        return
    
    if not phobert_results:
        print(f"Error: No PhoBERT results found in {args.phobert_output_dir}")
        return
    
    if not gpt_results:
        print(f"Error: No GPT results found in {args.gpt_output_dir}")
        return
    
    # Compare and display results
    compare_results(bert_std_results, phobert_results, gpt_results, args.output_file)
    
    # Save comparison to JSON
    comparison_data = {
        "bert_standard_results": bert_std_results,
        "phobert_results": phobert_results,
        "gpt_results": gpt_results,
        "comparison": {}
    }
    
    # Calculate differences for common metrics
    common_metrics = set(bert_std_results.keys()) & set(phobert_results.keys()) & set(gpt_results.keys())
    for metric in common_metrics:
        try:
            bert_std_val = bert_std_results[metric]
            phobert_val = phobert_results[metric]
            gpt_val = gpt_results[metric]
            
            comparison_data["comparison"][metric] = {
                "bert_standard": bert_std_val,
                "phobert": phobert_val,
                "gpt": gpt_val,
                "best_model": max([(bert_std_val, "BERT_Standard"), (phobert_val, "PhoBERT"), (gpt_val, "GPT")], key=lambda x: x[0])[1],
                "bert_std_vs_phobert": bert_std_val - phobert_val,
                "bert_std_vs_gpt": bert_std_val - gpt_val,
                "phobert_vs_gpt": phobert_val - gpt_val
            }
        except:
            pass
    
    with open(args.output_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\nComparison results saved to {args.output_file}")

if __name__ == "__main__":
    main() 
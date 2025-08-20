#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to run improved BERT triple classification experiments
"""

import subprocess
import os
import json
from datetime import datetime

def run_experiment(experiment_name, params):
    """Run a single experiment with given parameters"""
    
    output_dir = f"./output_KG_VNT_{experiment_name}/"
    
    cmd = [
        "python", "run_bert_triple_classifier.py",
        "--task_name", "kg",
        "--do_train",
        "--do_eval",
        "--do_predict",
        "--data_dir", "./data/KG_VNT",
        "--bert_model", "vinai/phobert-base",
        "--output_dir", output_dir,
        "--eval_batch_size", "512"
    ]
    
    # Add all parameters
    for key, value in params.items():
        if key != "experiment_name":
            cmd.extend([f"--{key}", str(value)])
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {experiment_name}")
    print(f"Parameters: {params}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)  # 4 hours timeout
        
        if result.returncode == 0:
            # Parse results
            eval_file = os.path.join(output_dir, "eval_results.txt")
            if os.path.exists(eval_file):
                with open(eval_file, 'r') as f:
                    lines = f.readlines()
                    results = {}
                    for line in lines:
                        if '=' in line:
                            key, value = line.strip().split(' = ')
                            try:
                                results[key] = float(value)
                            except:
                                results[key] = value
                    
                    params.update(results)
                    return params
            else:
                print(f"Eval results file not found: {eval_file}")
                return None
        else:
            print(f"Experiment failed with return code: {result.returncode}")
            print(f"Error: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"Experiment timed out after 4 hours")
        return None
    except Exception as e:
        print(f"Experiment error: {e}")
        return None

def main():
    # Define experiments
    experiments = {
        "baseline_improved": {
            "max_seq_length": 24,
            "train_batch_size": 32,
            "learning_rate": 2e-5,
            "num_train_epochs": 15.0,
            "gradient_accumulation_steps": 1,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "dropout": 0.1,
            "label_smoothing": 0.1,
            "patience": 3,
            "min_delta": 0.001
        },
        
        "focal_loss": {
            "max_seq_length": 24,
            "train_batch_size": 32,
            "learning_rate": 2e-5,
            "num_train_epochs": 15.0,
            "gradient_accumulation_steps": 1,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "dropout": 0.1,
            "focal_loss": True,
            "focal_alpha": 0.25,
            "focal_gamma": 2.0,
            "patience": 3,
            "min_delta": 0.001
        },
        
        "higher_lr": {
            "max_seq_length": 24,
            "train_batch_size": 32,
            "learning_rate": 3e-5,
            "num_train_epochs": 15.0,
            "gradient_accumulation_steps": 1,
            "warmup_steps": 200,
            "weight_decay": 0.01,
            "dropout": 0.1,
            "label_smoothing": 0.1,
            "patience": 3,
            "min_delta": 0.001
        },
        
        "longer_seq": {
            "max_seq_length": 32,
            "train_batch_size": 16,
            "learning_rate": 2e-5,
            "num_train_epochs": 15.0,
            "gradient_accumulation_steps": 2,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "dropout": 0.1,
            "label_smoothing": 0.1,
            "patience": 3,
            "min_delta": 0.001
        },
        
        "more_epochs": {
            "max_seq_length": 24,
            "train_batch_size": 32,
            "learning_rate": 2e-5,
            "num_train_epochs": 20.0,
            "gradient_accumulation_steps": 1,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "dropout": 0.1,
            "label_smoothing": 0.1,
            "patience": 5,
            "min_delta": 0.001
        },
        
        "stronger_regularization": {
            "max_seq_length": 24,
            "train_batch_size": 32,
            "learning_rate": 2e-5,
            "num_train_epochs": 15.0,
            "gradient_accumulation_steps": 1,
            "warmup_steps": 100,
            "weight_decay": 0.1,
            "dropout": 0.2,
            "label_smoothing": 0.2,
            "patience": 3,
            "min_delta": 0.001
        }
    }
    
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Starting experiments at {timestamp}")
    print(f"Total experiments: {len(experiments)}")
    
    for exp_name, params in experiments.items():
        result = run_experiment(exp_name, params)
        if result:
            results.append(result)
            print(f"✅ {exp_name} completed successfully")
            print(f"   Accuracy: {result.get('eval_accuracy', 'N/A'):.4f}")
            print(f"   F1 Score: {result.get('eval_f1_score', 'N/A'):.4f}")
        else:
            print(f"❌ {exp_name} failed")
    
    # Save all results
    results_file = f"experiment_results_{timestamp}.json"
    with open(results_file, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Find best result
    if results:
        best_result = max(results, key=lambda x: x.get('eval_accuracy', 0))
        print(f"\n{'='*60}")
        print(f"BEST RESULT:")
        print(f"Experiment: {best_result.get('experiment_name', 'Unknown')}")
        print(f"Accuracy: {best_result.get('eval_accuracy', 'N/A'):.4f}")
        print(f"F1 Score: {best_result.get('eval_f1_score', 'N/A'):.4f}")
        print(f"Precision: {best_result.get('eval_precision', 'N/A'):.4f}")
        print(f"Recall: {best_result.get('eval_recall', 'N/A'):.4f}")
        print(f"Loss: {best_result.get('eval_loss', 'N/A'):.4f}")
        print(f"{'='*60}")
        
        # Save best parameters
        best_params_file = f"best_parameters_{timestamp}.json"
        with open(best_params_file, "w", encoding='utf-8') as f:
            json.dump(best_result, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {results_file}")
        print(f"Best parameters saved to: {best_params_file}")
    
    # Print summary table
    if results:
        print(f"\n{'='*80}")
        print(f"SUMMARY TABLE:")
        print(f"{'='*80}")
        print(f"{'Experiment':<20} {'Accuracy':<10} {'F1 Score':<10} {'Precision':<10} {'Recall':<10}")
        print(f"{'-'*80}")
        
        for result in sorted(results, key=lambda x: x.get('eval_accuracy', 0), reverse=True):
            exp_name = result.get('experiment_name', 'Unknown')
            accuracy = result.get('eval_accuracy', 0)
            f1 = result.get('eval_f1_score', 0)
            precision = result.get('eval_precision', 0)
            recall = result.get('eval_recall', 0)
            
            print(f"{exp_name:<20} {accuracy:<10.4f} {f1:<10.4f} {precision:<10.4f} {recall:<10.4f}")

if __name__ == "__main__":
    main()

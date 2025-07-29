#!/usr/bin/env python3
"""
Demo script để test GPT Triple Classifier với một số examples đơn giản
"""

import os
import sys
from run_gpt_triple_classifier import GPTTripleClassifier

def demo_classifier():
    """Demo với một số triples đơn giản"""
    
    # Bạn cần thay thế bằng API key thật
    api_key = "your-openai-api-key-here"
    
    # Khởi tạo classifier
    classifier = GPTTripleClassifier(api_key, model="gpt-3.5-turbo")
    
    # Một số triples để test
    test_triples = [
        ("dog", "is_a", "animal"),
        ("dog", "is_a", "car"),
        ("Paris", "capital_of", "France"),
        ("Paris", "capital_of", "Germany"),
        ("water", "is_liquid", "true"),
        ("stone", "is_liquid", "true"),
        ("cat", "eats", "fish"),
        ("fish", "eats", "cat"),
    ]
    
    print("Testing GPT Triple Classifier...")
    print("=" * 50)
    
    for i, (head, relation, tail) in enumerate(test_triples, 1):
        print(f"\n{i}. Triple: {head} {relation} {tail}")
        
        try:
            prediction = classifier.classify_triple(head, relation, tail)
            result = "CORRECT" if prediction == 1 else "INCORRECT"
            print(f"   Prediction: {prediction} ({result})")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Delay để tránh rate limiting
        import time
        time.sleep(1)
    
    print("\n" + "=" * 50)
    print("Demo completed!")

def demo_with_sample_data():
    """Demo với dữ liệu mẫu từ dataset"""
    
    # Bạn cần thay thế bằng API key thật
    api_key = "your-openai-api-key-here"
    
    # Khởi tạo classifier
    classifier = GPTTripleClassifier(api_key, model="gpt-3.5-turbo")
    
    # Đọc một số examples từ dataset
    sample_triples = [
        ("__spiritual_bouquet_1", "_type_of", "__sympathy_card_1"),
        ("__nolo_contendere_1", "_type_of", "__sympathy_card_1"),
        ("__absorption_5", "_type_of", "__attention_4"),
        ("__sea_eagle_2", "_type_of", "__attention_4"),
        ("__avenge_1", "_type_of", "__penalise_1"),
        ("__b_t_u_1", "_type_of", "__penalise_1"),
    ]
    
    print("Testing with sample dataset triples...")
    print("=" * 60)
    
    for i, (head, relation, tail) in enumerate(sample_triples, 1):
        print(f"\n{i}. Triple: {head} {relation} {tail}")
        
        try:
            prediction = classifier.classify_triple(head, relation, tail)
            result = "CORRECT" if prediction == 1 else "INCORRECT"
            print(f"   Prediction: {prediction} ({result})")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Delay để tránh rate limiting
        import time
        time.sleep(1)
    
    print("\n" + "=" * 60)
    print("Dataset demo completed!")

if __name__ == "__main__":
    print("GPT Triple Classifier Demo")
    print("=" * 30)
    
    # Kiểm tra API key
    api_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
    
    if not api_key:
        print("No API key provided. Please set your API key in the script.")
        print("You can get one from: https://platform.openai.com/api-keys")
        sys.exit(1)
    
    # Cập nhật API key trong script
    with open(__file__, 'r') as f:
        content = f.read()
    
    content = content.replace("your-openai-api-key-here", api_key)
    
    with open(__file__, 'w') as f:
        f.write(content)
    
    print("\n1. Simple triples demo")
    print("2. Dataset triples demo")
    print("3. Both")
    
    choice = input("\nChoose demo type (1/2/3): ").strip()
    
    if choice == "1":
        demo_classifier()
    elif choice == "2":
        demo_with_sample_data()
    elif choice == "3":
        demo_classifier()
        print("\n" + "="*50)
        demo_with_sample_data()
    else:
        print("Invalid choice. Running simple demo...")
        demo_classifier() 
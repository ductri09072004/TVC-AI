#!/usr/bin/env python3
# coding=utf-8
"""
GPT-based Triple Classification using OpenAI API
This script performs triple classification using GPT API instead of BERT fine-tuning.
It reads the same data format as the BERT version and produces similar evaluation results.
"""

import argparse
import csv
import logging
import os
import json
import time
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import accuracy_score, classification_report
from openai import OpenAI
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPTTripleClassifier:
    """GPT-based triple classifier using OpenAI API"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", max_retries: int = 3):
        """
        Initialize the GPT classifier
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use
            max_retries: Maximum number of retries for API calls
        """
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.client = OpenAI(api_key=api_key)
        
        # System prompt for triple classification
        self.system_prompt = """You are a knowledge graph triple classifier. Your task is to determine whether a given triple (head entity, relation, tail entity) is correct or not.

Given a triple in the format: [HEAD_ENTITY] [RELATION] [TAIL_ENTITY]

You should respond with:
- "1" if the triple is correct/valid
- "0" if the triple is incorrect/invalid

Consider the semantic meaning and logical consistency of the triple. For example:
- "dog" "is_a" "animal" → 1 (correct)
- "dog" "is_a" "car" → 0 (incorrect)

Only respond with "1" or "0", nothing else."""

    def classify_triple(self, head: str, relation: str, tail: str) -> int:
        """
        Classify a single triple using GPT API
        
        Args:
            head: Head entity
            relation: Relation
            tail: Tail entity
            
        Returns:
            1 for correct, 0 for incorrect
        """
        triple_text = f"{head} {relation} {tail}"
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Classify this triple: {triple_text}"}
        ]
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=10,
                    temperature=0.0  # Deterministic responses
                )
                
                result = response.choices[0].message.content.strip()
                
                # Parse the response
                if result == "1":
                    return 1
                elif result == "0":
                    return 0
                else:
                    # Try to extract number from response
                    try:
                        return int(result)
                    except ValueError:
                        logger.warning(f"Unexpected response format: {result}")
                        # Default to 0 for unclear responses
                        return 0
                        
            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to classify triple after {self.max_retries} attempts")
                    return 0  # Default to 0 on failure

    def batch_classify(self, triples: List[Tuple[str, str, str]], batch_size: int = 10) -> List[int]:
        """
        Classify multiple triples in batches
        
        Args:
            triples: List of (head, relation, tail) tuples
            batch_size: Number of triples to process in parallel
            
        Returns:
            List of predictions (1 or 0)
        """
        predictions = []
        
        for i in tqdm(range(0, len(triples), batch_size), desc="Classifying triples"):
            batch = triples[i:i + batch_size]
            batch_predictions = []
            
            for head, relation, tail in batch:
                pred = self.classify_triple(head, relation, tail)
                batch_predictions.append(pred)
                
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
            
            predictions.extend(batch_predictions)
            
        return predictions


class DataProcessor:
    """Data processor for triple classification datasets"""
    
    def __init__(self):
        self.label_list = ["0", "1"]
    
    def _read_tsv(self, input_file: str) -> List[List[str]]:
        """Read TSV file"""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            lines = []
            for line in reader:
                lines.append(line)
        return lines
    
    def get_train_examples(self, data_dir: str) -> List[Tuple[str, str, str, int]]:
        """Get training examples"""
        return self._read_examples(os.path.join(data_dir, "train.txt"))
    
    def get_dev_examples(self, data_dir: str) -> List[Tuple[str, str, str, int]]:
        """Get development examples"""
        return self._read_examples(os.path.join(data_dir, "dev.txt"))
    
    def get_test_examples(self, data_dir: str) -> List[Tuple[str, str, str, int]]:
        """Get test examples"""
        return self._read_examples(os.path.join(data_dir, "test.txt"))
    
    def _read_examples(self, input_file: str) -> List[Tuple[str, str, str, int]]:
        """Read examples from TSV file"""
        lines = self._read_tsv(input_file)
        examples = []
        
        for (i, line) in enumerate(lines):
            if len(line) != 4:
                continue
                
            head = line[0].strip()
            relation = line[1].strip()
            tail = line[2].strip()
            label = int(line[3].strip())
            
            examples.append((head, relation, tail, label))
        
        return examples


def compute_metrics(predictions: List[int], labels: List[int]) -> Dict[str, float]:
    """
    Compute evaluation metrics
    
    Args:
        predictions: List of predicted labels
        labels: List of true labels
        
    Returns:
        Dictionary of metrics
    """
    accuracy = accuracy_score(labels, predictions)
    
    # Additional metrics
    report = classification_report(labels, predictions, output_dict=True)
    
    metrics = {
        "eval_accuracy": accuracy,
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1_score": report["weighted avg"]["f1-score"]
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    
    # Required parameters
    parser.add_argument("--data_dir", required=True, type=str,
                        help="The input data dir. Should contain the .tsv files")
    parser.add_argument("--output_dir", required=True, type=str,
                        help="The output directory where results will be written")
    parser.add_argument("--api_key", required=True, type=str,
                        help="OpenAI API key")
    
    # Optional parameters
    parser.add_argument("--model", default="gpt-3.5-turbo", type=str,
                        help="GPT model to use")
    parser.add_argument("--max_train_examples", type=int, default=None,
                        help="Maximum number of training examples to use")
    parser.add_argument("--max_eval_examples", type=int, default=None,
                        help="Maximum number of evaluation examples to use")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Batch size for API calls")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training evaluation")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run eval on the test set")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize processor and classifier
    processor = DataProcessor()
    classifier = GPTTripleClassifier(args.api_key, args.model)
    
    results = {}
    
    # Training evaluation
    if args.do_train:
        logger.info("***** Running training evaluation *****")
        train_examples = processor.get_train_examples(args.data_dir)
        
        if args.max_train_examples:
            train_examples = train_examples[:args.max_train_examples]
        
        logger.info(f"  Num examples = {len(train_examples)}")
        
        # Extract triples and labels
        train_triples = [(ex[0], ex[1], ex[2]) for ex in train_examples]
        train_labels = [ex[3] for ex in train_examples]
        
        # Get predictions
        train_predictions = classifier.batch_classify(train_triples, args.batch_size)
        
        # Compute metrics
        train_metrics = compute_metrics(train_predictions, train_labels)
        results.update({f"train_{k}": v for k, v in train_metrics.items()})
        
        logger.info(f"  Train accuracy = {train_metrics['eval_accuracy']:.4f}")
    
    # Development evaluation
    if args.do_eval:
        logger.info("***** Running evaluation *****")
        eval_examples = processor.get_dev_examples(args.data_dir)
        
        if args.max_eval_examples:
            eval_examples = eval_examples[:args.max_eval_examples]
        
        logger.info(f"  Num examples = {len(eval_examples)}")
        
        # Extract triples and labels
        eval_triples = [(ex[0], ex[1], ex[2]) for ex in eval_examples]
        eval_labels = [ex[3] for ex in eval_examples]
        
        # Get predictions
        eval_predictions = classifier.batch_classify(eval_triples, args.batch_size)
        
        # Compute metrics
        eval_metrics = compute_metrics(eval_predictions, eval_labels)
        results.update(eval_metrics)
        
        logger.info(f"  Eval accuracy = {eval_metrics['eval_accuracy']:.4f}")
        
        # Save detailed results
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(results.keys()):
                logger.info(f"  {key} = {results[key]}")
                writer.write(f"{key} = {results[key]}\n")
        
        # Save predictions
        output_pred_file = os.path.join(args.output_dir, "eval_predictions.tsv")
        with open(output_pred_file, "w", newline="", encoding="utf-8") as writer:
            csv_writer = csv.writer(writer, delimiter="\t")
            csv_writer.writerow(["head", "relation", "tail", "true_label", "predicted_label"])
            for i, (head, relation, tail) in enumerate(eval_triples):
                csv_writer.writerow([head, relation, tail, eval_labels[i], eval_predictions[i]])
    
    # Test prediction
    if args.do_predict:
        logger.info("***** Running prediction *****")
        test_examples = processor.get_test_examples(args.data_dir)
        
        if args.max_eval_examples:
            test_examples = test_examples[:args.max_eval_examples]
        
        logger.info(f"  Num examples = {len(test_examples)}")
        
        # Extract triples and labels
        test_triples = [(ex[0], ex[1], ex[2]) for ex in test_examples]
        test_labels = [ex[3] for ex in test_examples]
        
        # Get predictions
        test_predictions = classifier.batch_classify(test_triples, args.batch_size)
        
        # Compute metrics
        test_metrics = compute_metrics(test_predictions, test_labels)
        results.update({f"test_{k}": v for k, v in test_metrics.items()})
        
        logger.info(f"  Test accuracy = {test_metrics['eval_accuracy']:.4f}")
        
        # Save test results
        output_test_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_test_file, "w") as writer:
            logger.info("***** Test results *****")
            for key in sorted(results.keys()):
                logger.info(f"  {key} = {results[key]}")
                writer.write(f"{key} = {results[key]}\n")
        
        # Save test predictions
        output_test_pred_file = os.path.join(args.output_dir, "test_predictions.tsv")
        with open(output_test_pred_file, "w", newline="", encoding="utf-8") as writer:
            csv_writer = csv.writer(writer, delimiter="\t")
            csv_writer.writerow(["head", "relation", "tail", "true_label", "predicted_label"])
            for i, (head, relation, tail) in enumerate(test_triples):
                csv_writer.writerow([head, relation, tail, test_labels[i], test_predictions[i]])


if __name__ == "__main__":
    main() 
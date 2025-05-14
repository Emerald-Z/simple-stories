# compute_metrics_for_subsample.py
import json
import argparse
from datasets import load_dataset, load_from_disk
from diversity import compression_ratio, homogenization_score, ngram_diversity_score, extract_patterns, self_repetition_score
import os
from rouge_score import rouge_scorer
from evaluate import load
from tqdm import tqdm

def calculate_compression_ratio(text_string):
    return compression_ratio(text_string, 'gzip')

def calculate_homogenization_score(text_string, measure, dataset):
    if measure == 'rougel':
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    elif measure == 'bertscore':
        scorer = load("bertscore")
    elif measure == 'bleu':
        scorer = load("bleu")
    else:
        raise ValueError(f"Invalid measure: {measure}")
    
    preds = [x for j, x in enumerate(dataset) if j != text_string]
    refs = [text_string for _ in range(len(preds))]

    doc_score = 0.0
    if measure == 'rougel':
        doc_score = sum([scorer.score(p, text_string)['rougeL'].fmeasure for p in preds])
    elif measure == 'bertscore':
        computed_scores = scorer.compute(predictions=preds,
                                        references=refs,
                                        model_type="distilbert-base-uncased",
                                        batch_size=64)
        doc_score = sum(computed_scores['f1'])
    elif measure == 'bleu':
        bleu_batch_score = scorer.compute(predictions=preds,
                                    references=[[r] for r in refs])['bleu']
        doc_score = bleu_batch_score 
    return doc_score

def main():
    parser = argparse.ArgumentParser(description="Compute diversity metrics for a random subsample of a dataset.")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the dataset (local directory or Hugging Face ID).")
    parser.add_argument("--dataset_split", type=str, default="train",
                        help="Dataset split to use (e.g., 'train', 'test'). Default: 'train'.")
    parser.add_argument("--text_column", type=str, required=True,
                        help="Name of the column in the dataset containing the text data.")
    parser.add_argument("--num_subsamples", type=int, required=True,
                        help="Number of random samples to select from the dataset.")
    parser.add_argument("--output_json_path", type=str, required=True,
                        help="Path to save the output JSON file.", default="subsample_metrics.json")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling.")

    parser.add_argument("--hs_verbose", action='store_true', help="Enable verbose output for homogenization scores.")


    args = parser.parse_args()

    print(f"Loading dataset from: {args.dataset_path}")
    try:
        if os.path.exists(args.dataset_path) and os.path.isdir(args.dataset_path):
            dataset = load_from_disk(args.dataset_path)
        else:
            dataset = load_dataset(args.dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if args.dataset_split not in dataset:
        print(f"Error: Split '{args.dataset_split}' not found in dataset. Available splits: {list(dataset.keys())}")
        return
    
    split_data = dataset[args.dataset_split]

    if args.text_column not in split_data.column_names:
        print(f"Error: Text column '{args.text_column}' not found. Available columns: {split_data.column_names}")
        return

    num_available_samples = len(split_data)
    actual_num_subsamples = min(args.num_subsamples, num_available_samples)

    if actual_num_subsamples < args.num_subsamples:
        print(f"Warning: Requested {args.num_subsamples} samples, but only {num_available_samples} are available in split '{args.dataset_split}'. Using {actual_num_subsamples} samples.")
    
    if actual_num_subsamples == 0:
        print("No samples to process. Exiting.")
        # Create an empty JSON if needed
        with open(args.output_json_path, 'w') as f:
            json.dump({"samples": []}, f, indent=2)
        return

    print(f"Selecting {actual_num_subsamples} random samples with seed {args.seed}...")
    subsample_dataset = split_data.shuffle(seed=args.seed).select(range(actual_num_subsamples))
    
    subsampled_texts = subsample_dataset[args.text_column]

    print("Computing corpus-level homogenization scores for the subsample...")


    output_data = {"samples": []}
    print(f"\nProcessing {len(subsampled_texts)} texts for individual metrics...")

    for i, text in enumerate(tqdm(subsampled_texts, desc="Computing metrics", unit="text")):
        cr = calculate_compression_ratio(text)
        hs_bleu = calculate_homogenization_score(text, 'bleu', subsampled_texts)
        hs_rouge = calculate_homogenization_score(text, 'rougel', subsampled_texts)
        sample_entry = {
            # You might want to add an ID or truncate text if it's very long
            "original_text_id": f"subsample_item_{i}", 
            "text_preview": text[:100] + "..." if len(text) > 100 else text, # Example preview
            "scores": {
                "compression_ratio": cr,
                "homogenization_bleu_score": hs_bleu,
                "homogenization_rouge_score": hs_rouge
            }
        }
        output_data["samples"].append(sample_entry)

    print(f"\nWriting output to {args.output_json_path}")
    try:
        with open(args.output_json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print("Successfully wrote metrics to JSON.")
    except IOError as e:
        print(f"Error writing JSON to file: {e}")

if __name__ == "__main__":
    main()
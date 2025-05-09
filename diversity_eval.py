from datasets import load_from_disk
import argparse
from diversity import compression_ratio, homogenization_score, ngram_diversity_score, extract_patterns
import nltk
import json 
import os 
import numpy as np
import time # Import the time module
# comment out if already downloaded
# nltk.download('punkt_tab')

def compute_metrics(dataset):
    """Computes various diversity metrics for the given dataset."""
    metrics = {}
    metrics['compression_ratio_gzip'] = compression_ratio(dataset, 'gzip')

    # metrics['homogenization_score_rougel'] = homogenization_score(dataset, 'rougel')

    # Note: bertscore can be slow, uncomment if needed and installed
    # metrics['homogenization_score_bertscore'] = homogenization_score(dataset, 'bertscore')
    # metrics['homogenization_score_bleu'] = homogenization_score(dataset, 'bleu')
    return metrics

def compute_ngram_diversity(dataset, n):
    metrics = {}
    metrics['ngram_diversity_score'] = {}
    for i in range(1, n + 1):
        metrics['ngram_diversity_score'][i] = ngram_diversity_score(dataset, i)
    return metrics

def main():
    overall_start_time = time.time() # Start overall timer

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/simple_stories") # TODO: filepath
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--n", type=int, default=10, help="N-gram size for diversity and pattern extraction")
    parser.add_argument("--top_n", type=int, default=10, help="Top N patterns to extract")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--compute_metrics", action="store_true", help="Compute metrics")
    parser.add_argument("--compute_patterns_ngram", action="store_true", help="Compute n-gram patterns")
    args = parser.parse_args()

    print("Loading dataset...")
    load_start_time = time.time()
    dataset = load_from_disk(args.dataset)
    load_end_time = time.time()
    print(f"Dataset loaded in {load_end_time - load_start_time:.2f} seconds.")

    split = 'test' if 'simple_stories' in args.dataset else 'validation'
    # Ensure the dataset column is 'story' or adapt if needed
    if 'story' in dataset[split].column_names:
        text_column = 'story'
    else:
        text_column = dataset[split].column_names[0]

    # Limit samples if num_samples is less than the total dataset size
    if args.num_samples == -1:
        num_available_samples = len(dataset[split])
    else:
        num_available_samples = args.num_samples
    samples_to_take = min(args.num_samples, num_available_samples)
    if samples_to_take < args.num_samples:
        print(f"Warning: Requested {args.num_samples} samples, but only {num_available_samples} available in train split. Using {samples_to_take}.")

    data_example = dataset[split][text_column][:samples_to_take]
    dataset_name = args.dataset.split('/')[-1]
    n = args.n
    top_n = args.top_n

    if args.compute_metrics:
        print("\nStarting metric computation...")
        # Compute all metrics
        all_metrics = compute_metrics(data_example)
        print("Computed Metrics:")
        print(json.dumps(all_metrics, indent=4))

        ngram_metrics_start_time = time.time()
        ngram_metrics = compute_ngram_diversity(data_example, n)
        ngram_metrics_end_time = time.time()
        print(f"N-gram metrics computed in {ngram_metrics_end_time - ngram_metrics_start_time:.2f} seconds.")
        print("Computed N-gram Metrics:")
        print(json.dumps(ngram_metrics, indent=4))

        os.makedirs(args.output_dir, exist_ok=True)
        output_filename = os.path.join(args.output_dir, f"{dataset_name}_metrics_n{n}_samples{samples_to_take}.json")

        with open(output_filename, 'w') as f:
            json.dump(all_metrics, f, indent=4)

        ngram_output_filename = os.path.join(args.output_dir, f"{dataset_name}_ngram_metrics_n{n}_samples{samples_to_take}.json")
        with open(ngram_output_filename, 'w') as f:
            json.dump(ngram_metrics, f, indent=4)

        print(f"\nMetrics saved to: {output_filename}")
        print(f"N-gram metrics saved to: {ngram_output_filename}")


    if args.compute_patterns_ngram:
        print(f"\nExtracting top {top_n} POS patterns for n = 1 to {n}...")
        patterns_overall_start_time = time.time()

        pattern_output_dir = os.path.join(args.output_dir, f"{dataset_name}_patterns")
        os.makedirs(pattern_output_dir, exist_ok=True)

        for i in range(1, n + 1):
            print(f"  Extracting patterns for n={i}...")
            pattern_extraction_start_time = time.time()
            patterns = extract_patterns(data_example, i, top_n)
            patterns_serializable = {k: list(v) for k, v in patterns.items()}
            pattern_extraction_end_time = time.time()
            print(f"    Patterns for n={i} extracted in {pattern_extraction_end_time - pattern_extraction_start_time:.2f} seconds.")

            pattern_filename = os.path.join(
                pattern_output_dir,
                f"patterns_n{i}_top{top_n}_samples{samples_to_take}.json"
            )

            # Save the patterns to the specific file
            with open(pattern_filename, 'w') as f:
                json.dump(patterns_serializable, f, indent=4)
            print(f"    Patterns for n={i} saved to: {pattern_filename}")

        patterns_overall_end_time = time.time()
        print(f"\nAll pattern files extracted and saved in {patterns_overall_end_time - patterns_overall_start_time:.2f} seconds.")
        print(f"All pattern files saved in directory: {pattern_output_dir}")

    overall_end_time = time.time() # End overall timer
    print(f"\nTotal script execution time: {overall_end_time - overall_start_time:.2f} seconds.")


if __name__ == "__main__":
    main()

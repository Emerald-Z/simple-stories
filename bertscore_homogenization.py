from datasets import load_from_disk
import argparse
from diversity import compression_ratio, homogenization_score, ngram_diversity_score, extract_patterns
import nltk
import json 
import os 
import numpy as np
import time # Import the time module
from tqdm import tqdm
# comment out if already downloaded
# nltk.download('punkt_tab')

def compute_metrics(dataset):
    """Computes various diversity metrics for the given dataset."""
    metrics = {}
    #metrics['compression_ratio_gzip'] = compression_ratio(dataset, 'gzip')

    # metrics['homogenization_score_rougel'] = homogenization_score(dataset, 'rougel')

    # Note: bertscore can be slow, uncomment if needed and installed
    metrics['homogenization_score_bertscore'] = homogenization_score(dataset, 'bertscore')
    # metrics['homogenization_score_bleu'] = homogenization_score(dataset, 'bleu')
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

    num_subsets = 20
    subpopulation_size = 30 #size of subsample
    if args.compute_metrics:
        print("\nStarting metric computation...")
        # Compute all metrics
        scores = []
        for i in tqdm(range(num_subsets)):
            # sample 30 random points from the dataset
            indices = np.random.choice(len(data_example), subpopulation_size, replace=False)
            data_sample = [data_example[idx] for idx in indices]
            score = homogenization_score(data_sample, 'bertscore')
            scores.append(score)
        
        # Calculate average score and standard deviation
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Calculate 95% confidence interval using non-parametric bootstrapping
        n_bootstrap = 10000  # Number of bootstrap samples
        bootstrap_means = []
        for _ in range(n_bootstrap):
            # Sample with replacement from the original scores
            bootstrap_sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        # Calculate 95% confidence interval using percentiles
        confidence_interval = np.percentile(bootstrap_means, [2.5, 97.5])
        
        # Create metrics dictionary
        all_metrics = {
            'homogenization_score_bertscore': {
                'mean': float(avg_score),
                'std': float(std_score),
                'confidence_interval_95': [float(confidence_interval[0]), float(confidence_interval[1])],
                'samples': scores
            }
        }
        
        print("Computed Metrics:")
        print(json.dumps(all_metrics, indent=4))
        
        # Save scores to file
        os.makedirs(args.output_dir, exist_ok=True)
        output_filename = os.path.join(args.output_dir, f"{dataset_name}_bertscore_homogenization_samples{samples_to_take}.json")
        
        with open(output_filename, 'w') as f:
            json.dump(all_metrics, f, indent=4)
            
        print(f"\nBERTScore homogenization metrics saved to: {output_filename}")





if __name__ == "__main__":
    main()

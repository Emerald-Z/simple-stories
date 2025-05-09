import numpy as np
from diversity import compression_ratio, homogenization_score, ngram_diversity_score
from datasets import load_from_disk

def eval_with_paired_bootstrap(sys1, sys2,
                               num_samples=10000, sample_ratio=0.5,
                               eval_type='compression_ratio'):
    sys1_scores = []
    sys2_scores = []
    wins = [0, 0, 0] # [sys1 wins, sys2 wins, tie]
    n = min(len(sys1), len(sys2)) 
    ids = list(range(n))

    print(f"Starting paired bootstrap with {num_samples} samples for metric: {eval_type}")
    for i in range(num_samples):
        if (i + 1) % (num_samples // 10) == 0: 
            print(f"  Sample {i+1}/{num_samples}")
        # subsample the indices
        reduced_ids = np.random.choice(ids, int(n * sample_ratio), replace=True)

        # Create subsets of system outputs using the *same* indices
        reduced_sys1 = [sys1[i] for i in reduced_ids]
        reduced_sys2 = [sys2[i] for i in reduced_ids]

        if eval_type == 'compression_ratio':
            sys1_score = compression_ratio(reduced_sys1, 'gzip')
            sys2_score = compression_ratio(reduced_sys2, 'gzip')
        elif eval_type == 'homogenization_score_bleu':
            sys1_score = homogenization_score(reduced_sys1, 'bleu')
            sys2_score = homogenization_score(reduced_sys2, 'bleu')
        elif eval_type == 'homogenization_score_rouge':
            sys1_score = homogenization_score(reduced_sys1, 'rougel')
            sys2_score = homogenization_score(reduced_sys2, 'rougel')
        elif eval_type == 'ngram_diversity_3':
            sys1_score = ngram_diversity_score(reduced_sys1, 3)
            sys2_score = ngram_diversity_score(reduced_sys2, 3)

        # Compare scores (Higher score is assumed better by default)
        # Note: For metrics like compression ratio or homogenization, lower is better.
        if sys1_score > sys2_score:
            wins[0] += 1
        elif sys1_score < sys2_score:
            wins[1] += 1
        else:
            wins[2] += 1
        sys1_scores.append(sys1_score)
        sys2_scores.append(sys2_score)

    print("\nBootstrap results:")
    # Print win stats
    wins = [x / float(num_samples) for x in wins]
    print('Win ratio: sys1=%.3f, sys2=%.3f, tie=%.3f' % (wins[0], wins[1], wins[2]))

    # --- Significance Interpretation ---
    # Note: This assumes higher score = better. Adjust interpretation based on metric.
    if wins[0] > wins[1]:
        superiority = "sys1 superior"
        p_value = 1.0 - wins[0] # Probability sys2 is >= sys1
        # For lower-is-better metrics, this means sys1 had a lower score more often.
    elif wins[1] > wins[0]:
        superiority = "sys2 superior"
        p_value = 1.0 - wins[1] # Probability sys1 is >= sys2
        # For lower-is-better metrics, this means sys2 had a lower score more often.
    else:
        superiority = "No clear winner"
        p_value = wins[2] # Or calculate differently if needed, e.g., min(wins[0], wins[1])

    print(f'({superiority} with p=%.3f)' % p_value)
    print("Interpretation Note: For metrics like compression ratio or homogenization, a 'superior' system based on win ratio likely had *lower* scores more often.")
    # compression_ratio and homogenization are lower-is-better metrics
    # ngram_diversity is higher-is-better

    # Print system stats (Confidence Intervals)
    sys1_scores.sort()
    sys2_scores.sort()
    print('\nSystem Score Distributions:')
    print('sys1 mean=%.3f, median=%.3f, 95%% confidence interval=[%.3f, %.3f]' %
            (np.mean(sys1_scores), np.median(sys1_scores), sys1_scores[int(num_samples * 0.025)], sys1_scores[int(num_samples * 0.975)]))
    print('sys2 mean=%.3f, median=%.3f, 95%% confidence interval=[%.3f, %.3f]' %
            (np.mean(sys2_scores), np.median(sys2_scores), sys2_scores[int(num_samples * 0.025)], sys2_scores[int(num_samples * 0.975)]))

def main():
    tiny_stories = load_from_disk("data/tiny_stories")
    tiny_stories = tiny_stories['validation'][tiny_stories['validation'].column_names[0]]
    simple_stories = load_from_disk("data/simple_stories")
    simple_stories = simple_stories['test'][simple_stories['test'].column_names[0]]

    # Choose a diversity metric eval_type
    # Examples: 'diversity_compression_gzip', 'diversity_homogenization_bleu', 'diversity_ngram_3'
    metric_to_use = 'compression_ratio'

    # Run the bootstrap test
    eval_with_paired_bootstrap(simple_stories,
                            tiny_stories,
                            num_samples=10000, # Use fewer samples for faster testing initially
                            sample_ratio=0.8, # Sample a larger portion if desired
                            eval_type=metric_to_use)

if __name__ == "__main__":
    main()
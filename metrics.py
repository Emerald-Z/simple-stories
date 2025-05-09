from datasets import load_from_disk
from diversity import compression_ratio, homogenization_score, ngram_diversity_score, extract_patterns
from evaluate import load
from multiprocessing import Pool, cpu_count
from typing import List, Optional
from tqdm import tqdm
from rouge_score import rouge_scorer

import numpy as np
import time
import math 

# Worker function for parallel processing
def _calculate_partial_homogenization(args):
    """
    Worker function to calculate homogenization for a subset of reference documents.
    """
    sub_data_refs, full_data, measure, use_stemmer, model_name, batch_size = args

    if measure == 'rougel':
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=use_stemmer)
    elif measure == 'bertscore':
        # might be memory intensive
        scorer = load("bertscore")
    elif measure == 'bleu':
        scorer = load("bleu")
    else:
        raise ValueError("Scoring measure must be one of `rougel`, `bleu`, or `bertscore`.")

    partial_corpus_score = 0.0

    for ref_doc in sub_data_refs: # Iterate only over the assigned subset of references
        # Get all the other utterances from the full_data to compare against
        # Need to be careful if ref_doc itself is in full_data to exclude the exact instance
        # A safer way is to pass indices or ensure sub_data_refs are distinct items
        # For simplicity here, assuming ref_doc is one of the items in full_data
        preds = [x for x in full_data if x != ref_doc] # Simple exclusion
        if not preds:
            continue

        refs_for_preds = [ref_doc for _ in range(len(preds))]

        doc_score = 0.0
        if measure == 'rougel':
            doc_score = sum([scorer.score(p, ref_doc)['rougeL'].fmeasure for p in preds])
        elif measure == 'bertscore':
            # Ensure preds and refs_for_preds are not empty
            if preds and refs_for_preds:
                computed_scores = scorer.compute(predictions=preds,
                                                 references=refs_for_preds,
                                                 model_type=model_name,
                                                 batch_size=batch_size,
                                                 # device=device # If passing device explicitly
                                                 )
                doc_score = sum(computed_scores['f1'])
        elif measure == 'bleu':
            if preds and refs_for_preds:
                # BLEU expects references as list of lists
                computed_scores = scorer.compute(predictions=preds,
                                               references=[[r] for r in refs_for_preds])
                doc_score = computed_scores['bleu'] * len(preds) # BLEU returns an average, scale it back

        # Average score for the current ref_doc against all others
        if len(full_data) > 1: # Avoid division by zero if only one doc in original data
             partial_corpus_score += doc_score / (len(full_data) - 1)
        elif len(full_data) == 1 and not preds: # Only one document total
            # In this case, homogenization is maximal by some definitions.
            # The original code's logic `corpus_score += len(data)` if `corpus_score == 0`
            # and then `corpus_score / len(data)` would lead to 1.0.
            # If only one doc, it's perfectly homogenized with itself.
            # The loop for preds won't run.
            # Let's handle this at the aggregation stage based on original logic.
            pass


    return partial_corpus_score


def homogenization_score(
        data: List[str],
        measure: str = 'rougel',
        use_stemmer: Optional[bool] = False, 
        model: Optional[str] = "distilbert-base-uncased",
        verbose: Optional[bool] = True,
        batch_size: Optional[int] = 64,
        n_processes: Optional[int] = 1 # Number of processes to use
) -> float:
    """
    Structure borrowed from diversity package
     """
    if not data:
        return 0.0
    if len(data) == 1: # If only one document, it's perfectly homogenized.
        return 1.0

    if n_processes == 1 or len(data) < n_processes : #sequential
        if verbose:
            print(f"Running homogenization sequentially (data size: {len(data)})...")

        if measure == 'rougel':
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=use_stemmer)
        elif measure == 'bertscore':
            scorer = load("bertscore")
        elif measure == 'bleu':
            scorer = load("bleu")
        else:
            raise ValueError("Scoring measure must be one of `rougel`, `bleu`, or `bertscore`.")

        corpus_score_agg = 0.0
        
        iterator = tqdm(enumerate(data), total=len(data), disable=(not verbose)) if verbose else enumerate(data)
        if verbose:
             print('==> Scoring all pairs (sequential)')

        for i, ref_doc in iterator:
            preds = [x for j, x in enumerate(data) if j != i]
            if not preds: continue # Should only happen if data has 1 element, handled above

            refs_for_preds = [ref_doc for _ in range(len(preds))]
            
            doc_score = 0.0
            if measure == 'rougel':
                doc_score = sum([scorer.score(p, ref_doc)['rougeL'].fmeasure for p in preds])
            elif measure == 'bertscore':
                computed_scores = scorer.compute(predictions=preds,
                                               references=refs_for_preds,
                                               model_type=model,
                                               batch_size=batch_size)
                doc_score = sum(computed_scores['f1'])
            elif measure == 'bleu':
                bleu_batch_score = scorer.compute(predictions=preds,
                                           references=[[r] for r in refs_for_preds])['bleu']
                doc_score = bleu_batch_score 

            if len(data) > 1:
                corpus_score_agg += doc_score / (len(data) - 1)

    else: # parallel
        if verbose:
            print(f"Running homogenization in parallel with {n_processes} processes (data size: {len(data)})...")

        # Split data - each worker gets a chunk to process
        # full data still passed to each worker
        chunk_size = math.ceil(len(data) / n_processes)
        tasks_args = []
        for i in range(n_processes):
            start_index = i * chunk_size
            end_index = min((i + 1) * chunk_size, len(data))
            if start_index >= end_index:
                continue
            sub_data_refs_chunk = data[start_index:end_index]
            tasks_args.append((sub_data_refs_chunk, data, measure, use_stemmer, model, batch_size))

        with Pool(processes=n_processes) as pool:
            # Use tqdm for the main process to show progress over tasks
            results_iterator = pool.imap_unordered(_calculate_partial_homogenization, tasks_args)
            if verbose:
                results_iterator = tqdm(results_iterator, total=len(tasks_args), desc="Processing Chunks")
            
            partial_scores = list(results_iterator)

        corpus_score_agg = sum(partial_scores)

    # normalize
    if abs(corpus_score_agg) < 1e-9 and len(data) > 0 : # Check for effectively zero
        # original logic would make the final score 1.0.
        final_score = 1.0
    elif len(data) > 0:
        final_score = corpus_score_agg / len(data)
    else:
        final_score = 0.0

    return round(final_score, 3)

def main():
    dataset = load_from_disk("data/simple_stories")

    num_available_samples = 100
    samples_to_take = min(num_available_samples, len(dataset['test']['story']))

    data_example = dataset['test']['story'][:samples_to_take]

    start_time = time.time()
    print(homogenization_score(data_example, 'rougel', n_processes=8))
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


if __name__ == "__main__":
    main()


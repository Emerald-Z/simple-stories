import json
import argparse
from collections import defaultdict 

def load_patterns_with_counts(filepath):
    """Loads patterns and their counts from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        #structure is { "pattern_string": ["example1", "example2", ...] }
        # count is the number of examples (length of the list)
        patterns = {pattern: len(details) for pattern, details in data.items()}
        return patterns
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading {filepath}: {e}")
        return None

def print_counts(pattern_counts, title):
    """Prints the pattern counts in a sorted, formatted way."""
    print(f"\n--- {title} ---")
    if not pattern_counts:
        print("No patterns found or loaded.")
    else:
        sorted_patterns = sorted(pattern_counts.items())
        for pattern, count in sorted_patterns:
            print(f'"{pattern}": {count}')
    print("---------------------------------")


def main():
    parser = argparse.ArgumentParser(description="Find common patterns between two JSON files and show counts.")
    parser.add_argument("file1", help="Path to the first pattern JSON file.")
    parser.add_argument("file2", help="Path to the second pattern JSON file.")
    args = parser.parse_args()

    print(f"Loading patterns and counts from: {args.file1}")
    patterns1_counts = load_patterns_with_counts(args.file1)
    if patterns1_counts is None:
        return 

    print_counts(patterns1_counts, f"Pattern Counts for {args.file1}")

    print(f"\nLoading patterns and counts from: {args.file2}")
    patterns2_counts = load_patterns_with_counts(args.file2)
    if patterns2_counts is None:
        return 

    print_counts(patterns2_counts, f"Pattern Counts for {args.file2}")

    patterns1_set = set(patterns1_counts.keys())
    patterns2_set = set(patterns2_counts.keys())
    common_patterns = patterns1_set.intersection(patterns2_set)

    print(f"\n--- Common Patterns Found in Both Files ---")
    if not common_patterns:
        print("No common patterns found between the two files.")
    else:
        sorted_common_patterns = sorted(list(common_patterns))
        for pattern in sorted_common_patterns:
            # print(f'"{pattern}" (File1: {patterns1_counts[pattern]}, File2: {patterns2_counts[pattern]})')
            print(f'"{pattern}"') # Just print the common pattern string
    print("-------------------------------------------")


if __name__ == "__main__":
    main()

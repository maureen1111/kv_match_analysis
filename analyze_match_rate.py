import json
import hashlib
import pickle
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

# NONE_HASH for first block (following vLLM's approach)
NONE_HASH = hashlib.sha256(b"NONE_HASH_SEED").digest()


def hash_block_tokens(
    parent_block_hash: Optional[bytes],
    curr_block_token_ids: List[int]
) -> bytes:
    """
    Compute hash for a block of tokens following vLLM's approach.

    Args:
        parent_block_hash: The hash of the parent block. None if this is the first block.
        curr_block_token_ids: A list of token ids in the current block.

    Returns:
        The hash value (bytes) of the block.
    """
    if parent_block_hash is None:
        parent_block_hash = NONE_HASH

    curr_block_token_ids_tuple = tuple(curr_block_token_ids)

    input_data = (parent_block_hash, curr_block_token_ids_tuple)
    input_bytes = pickle.dumps(input_data, protocol=pickle.HIGHEST_PROTOCOL)

    return hashlib.sha256(input_bytes).digest()


def bytes_to_hex(hash_bytes: bytes) -> str:
    """Convert bytes hash to hex string for display."""
    return hash_bytes.hex()


def extract_requests(file_path: str) -> List[Dict]:
    """Extract requests from the JSON file."""
    print(f"Loading requests from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    requests = []
    for session in data['data']:
        for payload in session['payloads']:
            for message in payload['messages']:
                if 'content' in message:
                    requests.append({
                        'session_id': session['session_id'],
                        'content': message['content']
                    })

    print(f"Extracted {len(requests)} requests")
    return requests


def tokenize_and_block(requests: List[Dict], tokenizer, block_size: int) -> Tuple[List[Tuple[str, int, int]], Dict, int]:
    """
    Tokenize requests and create blocks.

    Returns:
        - List of (block_hash_hex, request_idx, block_position) tuples in sequential order
        - Dict mapping block_hash_hex to list of (request_idx, block_position, global_position)
        - Total number of tokens processed
    """
    print("Tokenizing requests and creating blocks...")

    all_blocks = [] 
    block_occurrences = defaultdict(list) 

    global_token_count = 0

    for req_idx, request in enumerate(requests):
        if req_idx % 100 == 0:
            print(f"Processing request {req_idx}/{len(requests)}...")

        content = request['content']

        token_ids = tokenizer.encode(content, add_special_tokens=False)

        num_blocks = len(token_ids) // block_size
        parent_block_hash = None  # first block

        for block_idx in range(num_blocks):
            start_idx = block_idx * block_size
            end_idx = start_idx + block_size
            block_tokens = token_ids[start_idx:end_idx]

            # using vLLM's approach (with parent hash)
            block_hash_bytes = hash_block_tokens(parent_block_hash, block_tokens)
            block_hash_hex = bytes_to_hex(block_hash_bytes)

            all_blocks.append((block_hash_hex, req_idx, block_idx))
            block_occurrences[block_hash_hex].append((req_idx, block_idx, global_token_count))

            parent_block_hash = block_hash_bytes

            global_token_count += block_size

    print(f"Total blocks created: {len(all_blocks)}")
    print(f"Unique blocks: {len(block_occurrences)}")
    print(f"Total tokens processed: {global_token_count}")

    return all_blocks, block_occurrences, global_token_count


def analyze_reuse_intervals(all_blocks: List[Tuple[str, int, int]],
                            block_occurrences: Dict,
                            block_size: int) -> List[int]:
    """

    For each block that reappears, calculate the number of tokens from OTHER requests
    between appearances.

    Returns:
        List of intervals (in tokens) for each block reuse
    """
    print("Analyzing reuse intervals...")

    intervals = []

    block_positions = {}  # (block_hash, occurrence_idx) -> (global_token_pos, request_idx)
    block_occurrence_counter = defaultdict(int)  

    global_token_pos = 0
    for block_hash, req_idx, block_idx in tqdm(all_blocks, total=len(all_blocks)):

        occurrence_idx = block_occurrence_counter[block_hash]
        block_occurrence_counter[block_hash] += 1

        block_positions[(block_hash, occurrence_idx)] = (global_token_pos, req_idx)
        global_token_pos += block_size

    # Now analyze intervals
    for block_hash, occurrences in block_occurrences.items():
        if len(occurrences) < 2:
            continue  

        for i in range(1, len(occurrences)):
            prev_req_idx, prev_block_idx, prev_global_pos = occurrences[i-1]
            curr_req_idx, curr_block_idx, curr_global_pos = occurrences[i]

            total_tokens_between = curr_global_pos - (prev_global_pos + block_size)

            if total_tokens_between >= 0:
                # Count tokens from other requests
                if curr_req_idx != prev_req_idx:
                    intervals.append(total_tokens_between)

    print(f"Found {len(intervals)} block reuse instances")
    return intervals


def plot_distribution(intervals: List[int], 
                     block_size: int,
                     total_tokens_processed: int,
                     gpu_kv_cache_size: Optional[int] = None,
                     output_file: str = "block_reuse_distribution.png"):
    """
    Create a scatter plot showing the complementary cumulative distribution.

    X-axis: interval (number of tokens from other requests, log10 scale)
    Y-axis: percentage of TOKENS (from all processed tokens) that did NOT reappear within x tokens (i.e., interval > x)

    """
    print("Generating plot...")

    if not intervals:
        print("No intervals to plot!")
        return

    # Filter out intervals <= 0 
    valid_intervals = [interval for interval in intervals if interval > 0]

    if not valid_intervals:
        print("No valid intervals (>0) to plot!")
        return

    # Sort intervals
    sorted_intervals = sorted(valid_intervals)
    all_sorted = sorted(intervals)  # Keep all intervals for statistics

    # CCDF based on TOKENS
    # percentages[i] = percentage of TOKENS (from all processed tokens) with interval > sorted_intervals[i]
    n_valid_blocks = len(sorted_intervals)
    n_total_blocks = len(intervals)
    
    # For each point, calculate how many tokens have interval > x
    # Denominator is all tokens processed in all requests
    percentages = []
    for i in range(n_valid_blocks):
        num_blocks_greater = n_valid_blocks - (i + 1)
        num_tokens_greater = num_blocks_greater * block_size
        percentage = (num_tokens_greater / total_tokens_processed) * 100
        percentages.append(percentage)

    plt.figure(figsize=(12, 8))

    scatter = plt.scatter(sorted_intervals, percentages, alpha=0.5, s=1, label='Block reuse (CCDF)')

    plt.xscale('log', base=10)

    if gpu_kv_cache_size is not None and gpu_kv_cache_size > 0:
        if gpu_kv_cache_size >= min(sorted_intervals) and gpu_kv_cache_size <= max(sorted_intervals):
            # Find the closest interval value
            idx = np.searchsorted(sorted_intervals, gpu_kv_cache_size)
            if idx < len(percentages):
                y_value = percentages[idx]
            else:
                y_value = percentages[-1]
    
            plt.axvline(x=gpu_kv_cache_size, color='red', linestyle='--', linewidth=2, 
                       label=f'GPU KV Cache Size ({gpu_kv_cache_size:,} tokens)')
            
            plt.axhline(y=y_value, color='red', linestyle='--', linewidth=2, alpha=0.7)
            
            plt.plot(gpu_kv_cache_size, y_value, 'ro', markersize=8, zorder=5)
            
            plt.text(gpu_kv_cache_size * 1.2, y_value, f'{y_value:.1f}%', 
                    fontsize=10, color='red', verticalalignment='center')

    plt.xlabel('Interval (tokens from other requests, log10 scale)', fontsize=12)
    plt.ylabel('Percentage of tokens without reuse within interval (%)', fontsize=12)
    plt.title(f'KV Cache Block Reuse Distribution (block_size={block_size})', fontsize=14)

    plt.grid(True, alpha=0.3, which='both') 

    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)

    zero_intervals = sum(1 for x in intervals if x == 0)
    negative_intervals = sum(1 for x in intervals if x < 0)

    stats_text = f"""Statistics:
Total tokens processed: {total_tokens_processed:,}
Total reuse instances (blocks): {n_total_blocks}
  - Plotted (>0): {n_valid_blocks}
  - Zero intervals: {zero_intervals}
  - Negative intervals: {negative_intervals}

Interval statistics:
  Min: {min(all_sorted):,}
  Max: {max(all_sorted):,}
  Median: {all_sorted[n_total_blocks//2]:,}
  Mean: {np.mean(intervals):.0f}
"""
    plt.text(0.02, 0.35, stats_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

    print(f"\nFiltered out {zero_intervals} zero intervals and {negative_intervals} negative intervals from plot")
    print(f"Total tokens processed: {total_tokens_processed:,}")
    print(f"Reused tokens (plotted): {n_valid_blocks * block_size:,} ({n_valid_blocks * block_size / total_tokens_processed * 100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze KV cache block reuse patterns from request logs'
    )
    parser.add_argument('--block-size', type=int, default=64,
                       help='Block size for KV cache (default: 64)')
    parser.add_argument('--input-file', type=str, 
                       default='/workspace/inputs_conversation.json',
                       help='Path to input JSON file')
    parser.add_argument('--model-name', type=str,
                       default='deepseek-ai/DeepSeek-V3.1',
                       help='Model name or path for tokenizer')
    parser.add_argument('--gpu-kv-cache-size', type=int, default=None,
                       help='GPU KV cache size in tokens (optional, will draw reference lines if set)')
    parser.add_argument('--output', type=str, default='block_reuse_distribution.png',
                       help='Output file path for the plot')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("KV Cache Block Reuse Analysis")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Block size: {args.block_size}")
    print(f"  Input file: {args.input_file}")
    print(f"  Model name: {args.model_name}")
    print(f"  GPU KV cache size: {args.gpu_kv_cache_size if args.gpu_kv_cache_size else 'Not set'}")
    print(f"  Output file: {args.output}")

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    requests = extract_requests(args.input_file)

    all_blocks, block_occurrences, total_tokens = tokenize_and_block(requests, tokenizer, args.block_size)

    intervals = analyze_reuse_intervals(all_blocks, block_occurrences, args.block_size)

    plot_distribution(intervals, args.block_size, total_tokens, args.gpu_kv_cache_size, args.output)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Deep debug script to thoroughly inspect SmolVLM H5 files
"""

import h5py
import numpy as np
import sys
from pathlib import Path

def debug_h5_file(h5_path):
    """Perform deep inspection of H5 file"""
    print(f"\n{'='*80}")
    print(f"DEEP DEBUG: {h5_path.name}")
    print(f"{'='*80}")

    with h5py.File(h5_path, 'r') as f:
        # 1. List all top-level keys
        print(f"\n📁 Top-level keys: {list(f.keys())}")

        # 2. Check number of samples
        sample_ids = list(f.keys())
        print(f"\n📊 Total samples in file: {len(sample_ids)}")

        # 3. Inspect first sample structure
        if sample_ids:
            first_sample = sample_ids[0]
            print(f"\n🔍 Inspecting sample: {first_sample}")
            print(f"   Keys in sample: {list(f[first_sample].keys())}")

            # 4. Check vision_only_representation
            if 'vision_only_representation' in f[first_sample]:
                vision_data = f[first_sample]['vision_only_representation'][:]
                print(f"\n✓ vision_only_representation:")
                print(f"   Shape: {vision_data.shape}")
                print(f"   Dtype: {vision_data.dtype}")
                print(f"   Min: {vision_data.min():.4f}, Max: {vision_data.max():.4f}, Mean: {vision_data.mean():.4f}")
                print(f"   Contains NaN: {np.isnan(vision_data).any()}")
                print(f"   Contains Inf: {np.isinf(vision_data).any()}")
            else:
                print(f"\n✗ vision_only_representation: MISSING")

            # 5. Check vision_token_representation
            if 'vision_token_representation' in f[first_sample]:
                vision_token_group = f[first_sample]['vision_token_representation']
                layers = list(vision_token_group.keys())
                print(f"\n✓ vision_token_representation:")
                print(f"   Number of layers: {len(layers)}")
                print(f"   Layer names: {sorted(layers)}")

                for layer in sorted(layers):
                    layer_data = vision_token_group[layer][:]
                    print(f"   - {layer}: shape={layer_data.shape}, dtype={layer_data.dtype}, "
                          f"mean={layer_data.mean():.4f}, has_nan={np.isnan(layer_data).any()}")
            else:
                print(f"\n✗ vision_token_representation: MISSING")

            # 6. Check query_token_representation
            if 'query_token_representation' in f[first_sample]:
                query_token_group = f[first_sample]['query_token_representation']
                layers = list(query_token_group.keys())
                print(f"\n✓ query_token_representation:")
                print(f"   Number of layers: {len(layers)}")
                print(f"   Layer names: {sorted(layers)}")

                for layer in sorted(layers):
                    layer_data = query_token_group[layer][:]
                    print(f"   - {layer}: shape={layer_data.shape}, dtype={layer_data.dtype}, "
                          f"mean={layer_data.mean():.4f}, has_nan={np.isnan(layer_data).any()}")
            else:
                print(f"\n✗ query_token_representation: MISSING")

            # 7. Check metadata attributes
            print(f"\n📋 Metadata attributes:")
            if 'image_id' in f[first_sample].attrs:
                print(f"   ✓ image_id: {f[first_sample].attrs['image_id']}")
            else:
                print(f"   ✗ image_id: MISSING")

            if 'question' in f[first_sample].attrs:
                question = f[first_sample].attrs['question']
                print(f"   ✓ question: {question[:100]}..." if len(question) > 100 else f"   ✓ question: {question}")
            else:
                print(f"   ✗ question: MISSING")

            if 'ground_truth_answer' in f[first_sample].attrs:
                gt_answer = f[first_sample].attrs['ground_truth_answer']
                print(f"   ✓ ground_truth_answer: {gt_answer}")
            else:
                print(f"   ✗ ground_truth_answer: MISSING")

            if 'answer' in f[first_sample].attrs:
                answer = f[first_sample].attrs['answer']
                print(f"   ✓ answer: {answer[:100]}..." if len(answer) > 100 else f"   ✓ answer: {answer}")
            else:
                print(f"   ✗ answer: MISSING")

        # 8. Sample multiple entries to check consistency
        print(f"\n🔬 Sampling 5 random entries for consistency check:")
        import random
        sample_indices = random.sample(range(len(sample_ids)), min(5, len(sample_ids)))

        for idx in sample_indices:
            sample_id = sample_ids[idx]
            sample = f[sample_id]

            vision_only_shape = sample['vision_only_representation'].shape if 'vision_only_representation' in sample else None
            vision_token_layers = len(list(sample['vision_token_representation'].keys())) if 'vision_token_representation' in sample else 0
            query_token_layers = len(list(sample['query_token_representation'].keys())) if 'query_token_representation' in sample else 0

            has_image_id = 'image_id' in sample.attrs
            has_question = 'question' in sample.attrs
            has_gt_answer = 'ground_truth_answer' in sample.attrs
            has_answer = 'answer' in sample.attrs

            print(f"   Sample {sample_id}:")
            print(f"      vision_only: {vision_only_shape}, "
                  f"vision_layers: {vision_token_layers}, "
                  f"query_layers: {query_token_layers}")
            print(f"      metadata: image_id={has_image_id}, question={has_question}, "
                  f"gt_answer={has_gt_answer}, answer={has_answer}")

        # 9. Check for any null/empty ground truth answers
        print(f"\n🔎 Checking for null/empty ground truth answers:")
        null_count = 0
        na_count = 0

        for sample_id in sample_ids[:100]:  # Check first 100
            sample = f[sample_id]
            if 'ground_truth_answer' in sample.attrs:
                gt_answer = sample.attrs['ground_truth_answer']
                if gt_answer == "N/A":
                    na_count += 1
                elif not gt_answer or gt_answer.strip() == "":
                    null_count += 1

        print(f"   Checked {min(100, len(sample_ids))} samples:")
        print(f"   - 'N/A' values: {na_count}")
        print(f"   - Empty/null values: {null_count}")

        # 10. Verify expected shapes
        print(f"\n✅ Expected Shapes Validation:")
        if sample_ids:
            sample = f[sample_ids[0]]

            # Vision only should be (1152,)
            if 'vision_only_representation' in sample:
                shape = sample['vision_only_representation'].shape
                expected = (1152,)
                status = "✓" if shape == expected else "✗"
                print(f"   {status} vision_only_representation: {shape} (expected {expected})")

            # Vision token layers should be (2048,)
            if 'vision_token_representation' in sample:
                for layer in sample['vision_token_representation'].keys():
                    shape = sample['vision_token_representation'][layer].shape
                    expected = (2048,)
                    status = "✓" if shape == expected else "✗"
                    print(f"   {status} vision_token/{layer}: {shape} (expected {expected})")

            # Query token layers should be (2048,)
            if 'query_token_representation' in sample:
                for layer in sample['query_token_representation'].keys():
                    shape = sample['query_token_representation'][layer].shape
                    expected = (2048,)
                    status = "✓" if shape == expected else "✗"
                    print(f"   {status} query_token/{layer}: {shape} (expected {expected})")

def main():
    output_dir = Path("/root/akhil/HALP_EACL_Models/Models/Smol_VL/smolvlm_output")
    h5_files = sorted(output_dir.glob("*.h5"))

    if not h5_files:
        print("❌ No H5 files found!")
        sys.exit(1)

    print(f"\n🔍 Found {len(h5_files)} H5 files")
    print(f"Files: {[f.name for f in h5_files]}")

    # Deep inspect first and last file
    print(f"\n{'#'*80}")
    print(f"# INSPECTING FIRST FILE")
    print(f"{'#'*80}")
    debug_h5_file(h5_files[0])

    if len(h5_files) > 1:
        print(f"\n{'#'*80}")
        print(f"# INSPECTING LAST FILE")
        print(f"{'#'*80}")
        debug_h5_file(h5_files[-1])

    # Summary across all files
    print(f"\n{'='*80}")
    print(f"SUMMARY ACROSS ALL FILES")
    print(f"{'='*80}")

    total_samples = 0
    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f:
            num_samples = len(list(f.keys()))
            total_samples += num_samples
            print(f"   {h5_file.name}: {num_samples} samples")

    print(f"\n   Total samples across all files: {total_samples}")
    print(f"\n✅ Deep debug complete!")

if __name__ == "__main__":
    main()

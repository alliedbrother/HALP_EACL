#!/usr/bin/env python3
"""
Comprehensive validation of SmolVLM H5 files against expected structure
"""

import h5py
import numpy as np
from pathlib import Path

def validate_sample_structure(sample, sample_id):
    """Validate a single sample against expected structure"""
    issues = []
    warnings = []

    # 1. Check required metadata datasets
    required_metadata = ['image_id', 'question', 'ground_truth_answer', 'answer']
    for key in required_metadata:
        if key not in sample:
            issues.append(f"Missing metadata: {key}")
        else:
            # Check it's a scalar dataset
            if sample[key].shape != ():
                issues.append(f"{key} should be scalar, got shape {sample[key].shape}")

    # 2. Check vision_only_representation
    if 'vision_only_representation' not in sample:
        issues.append("Missing vision_only_representation")
    else:
        vision_only = sample['vision_only_representation']
        expected_shape = (1152,)
        if vision_only.shape != expected_shape:
            issues.append(f"vision_only shape: expected {expected_shape}, got {vision_only.shape}")

        # Check for data quality
        data = vision_only[:]
        if np.isnan(data).any():
            issues.append("vision_only contains NaN values")
        if np.isinf(data).any():
            issues.append("vision_only contains Inf values")

    # 3. Check vision_token_representation
    if 'vision_token_representation' not in sample:
        issues.append("Missing vision_token_representation group")
    else:
        vision_token = sample['vision_token_representation']
        expected_layers = ['layer_0', 'layer_6', 'layer_12', 'layer_18', 'layer_23']

        if not isinstance(vision_token, h5py.Group):
            issues.append("vision_token_representation should be a group")
        else:
            actual_layers = sorted(list(vision_token.keys()))
            if sorted(expected_layers) != actual_layers:
                warnings.append(f"vision_token layers: expected {sorted(expected_layers)}, got {actual_layers}")

            for layer in vision_token.keys():
                layer_data = vision_token[layer]
                expected_shape = (2048,)
                if layer_data.shape != expected_shape:
                    issues.append(f"vision_token/{layer} shape: expected {expected_shape}, got {layer_data.shape}")

                # Check for data quality
                data = layer_data[:]
                if np.isnan(data).any():
                    issues.append(f"vision_token/{layer} contains NaN values")
                if np.isinf(data).any():
                    issues.append(f"vision_token/{layer} contains Inf values")

    # 4. Check query_token_representation
    if 'query_token_representation' not in sample:
        issues.append("Missing query_token_representation group")
    else:
        query_token = sample['query_token_representation']
        expected_layers = ['layer_0', 'layer_6', 'layer_12', 'layer_18', 'layer_23']

        if not isinstance(query_token, h5py.Group):
            issues.append("query_token_representation should be a group")
        else:
            actual_layers = sorted(list(query_token.keys()))
            if sorted(expected_layers) != actual_layers:
                warnings.append(f"query_token layers: expected {sorted(expected_layers)}, got {actual_layers}")

            for layer in query_token.keys():
                layer_data = query_token[layer]
                expected_shape = (2048,)
                if layer_data.shape != expected_shape:
                    issues.append(f"query_token/{layer} shape: expected {expected_shape}, got {layer_data.shape}")

                # Check for data quality
                data = layer_data[:]
                if np.isnan(data).any():
                    issues.append(f"query_token/{layer} contains NaN values")
                if np.isinf(data).any():
                    issues.append(f"query_token/{layer} contains Inf values")

    return issues, warnings

def main():
    output_dir = Path("/root/akhil/HALP_EACL_Models/Models/Smol_VL/smolvlm_output")
    h5_files = sorted(output_dir.glob("*.h5"))

    print("="*80)
    print("COMPREHENSIVE H5 FILE VALIDATION")
    print("="*80)

    total_samples = 0
    total_issues = 0
    total_warnings = 0
    file_summaries = []

    for h5_file in h5_files:
        print(f"\n📁 Validating: {h5_file.name}")

        with h5py.File(h5_file, 'r') as f:
            sample_ids = list(f.keys())
            num_samples = len(sample_ids)
            total_samples += num_samples

            file_issues = 0
            file_warnings = 0

            # Validate first 5 samples thoroughly
            for idx, sample_id in enumerate(sample_ids[:5]):
                sample = f[sample_id]
                issues, warnings = validate_sample_structure(sample, sample_id)

                if issues:
                    print(f"   ✗ {sample_id}: {len(issues)} issues")
                    for issue in issues:
                        print(f"      - {issue}")
                    file_issues += len(issues)

                if warnings:
                    for warning in warnings:
                        print(f"      ! {warning}")
                    file_warnings += len(warnings)

            # Quick check on remaining samples
            for sample_id in sample_ids[5:]:
                sample = f[sample_id]
                issues, warnings = validate_sample_structure(sample, sample_id)
                file_issues += len(issues)
                file_warnings += len(warnings)

            total_issues += file_issues
            total_warnings += file_warnings

            status = "✅ PASS" if file_issues == 0 else "❌ FAIL"
            print(f"   {status}: {num_samples} samples, {file_issues} issues, {file_warnings} warnings")

            file_summaries.append({
                'file': h5_file.name,
                'samples': num_samples,
                'issues': file_issues,
                'warnings': file_warnings
            })

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    for summary in file_summaries:
        status = "✅" if summary['issues'] == 0 else "❌"
        print(f"{status} {summary['file']}: {summary['samples']} samples, "
              f"{summary['issues']} issues, {summary['warnings']} warnings")

    print(f"\n📊 Overall Statistics:")
    print(f"   Total files: {len(h5_files)}")
    print(f"   Total samples: {total_samples}")
    print(f"   Total issues: {total_issues}")
    print(f"   Total warnings: {total_warnings}")

    if total_issues == 0:
        print(f"\n✅ ALL VALIDATION CHECKS PASSED!")
        print(f"\n✓ All H5 files contain correct structure:")
        print(f"   - vision_only_representation: (1152,)")
        print(f"   - vision_token_representation: 5 layers × (2048,)")
        print(f"   - query_token_representation: 5 layers × (2048,)")
        print(f"   - Metadata: image_id, question, ground_truth_answer, answer")
        print(f"   - No NaN or Inf values detected")
    else:
        print(f"\n❌ VALIDATION FAILED with {total_issues} issues")

if __name__ == "__main__":
    main()

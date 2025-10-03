#!/usr/bin/env python3
"""
Build Hallucination Dataset CSV from Smol VLM HDF5 Files

NOTE: Smol HDF5 files do not contain ground_truth_answer field.
We'll extract question_id, image_id, question, and model_answer from HDF5,
then merge with ground truth from a reference dataset (Gemma3).
"""

import h5py
import pandas as pd
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import glob


def extract_data_from_h5(h5_path):
    """Extract question data from a single HDF5 file"""
    data_rows = []

    try:
        with h5py.File(h5_path, 'r') as f:
            # Iterate through all question_id groups
            for key in f.keys():
                try:
                    # Remove 'question_id_' prefix if present
                    question_id = key.replace('question_id_', '')

                    q_group = f[key]

                    # Extract text fields
                    question = q_group['question'][()].decode('utf-8') if 'question' in q_group else ''
                    image_id = q_group['image_id'][()].decode('utf-8') if 'image_id' in q_group else ''
                    model_answer = q_group['answer'][()].decode('utf-8') if 'answer' in q_group else ''

                    data_rows.append({
                        'question_id': question_id,
                        'image_id': image_id,
                        'question': question,
                        'model_answer': model_answer
                    })

                except Exception as e:
                    print(f"  Warning: Failed to extract {key}: {e}")
                    continue

    except Exception as e:
        print(f"Error reading {h5_path}: {e}")
        return []

    return data_rows


def load_ground_truth_reference(reference_csv_path):
    """Load ground truth from a reference dataset (e.g., Gemma3)"""
    print(f"📖 Loading ground truth reference from: {reference_csv_path}")

    ref_df = pd.read_csv(reference_csv_path)

    # Create mapping: question_id -> ground_truth_answer
    ground_truth_map = dict(zip(ref_df['question_id'], ref_df['ground_truth_answer']))

    print(f"   ✓ Loaded {len(ground_truth_map)} ground truth answers")

    return ground_truth_map


def build_hallucination_dataset(input_dir, output_csv, reference_csv):
    """Build hallucination dataset from all .h5 files in directory"""

    print("=" * 60)
    print("Building Hallucination Dataset for SmolVLM")
    print("=" * 60)

    # Find all .h5 files
    h5_pattern = os.path.join(input_dir, "*.h5")
    h5_files = sorted(glob.glob(h5_pattern))

    if not h5_files:
        print(f"❌ No .h5 files found in {input_dir}")
        return

    print(f"📁 Found {len(h5_files)} HDF5 files")
    print()

    # Load ground truth reference
    ground_truth_map = load_ground_truth_reference(reference_csv)
    print()

    # Extract data from all files
    all_data = []

    for h5_file in tqdm(h5_files, desc="Processing HDF5 files"):
        filename = os.path.basename(h5_file)
        print(f"\n📄 Processing: {filename}")

        data_rows = extract_data_from_h5(h5_file)
        all_data.extend(data_rows)

        print(f"   ✓ Extracted {len(data_rows)} samples")

    print()
    print("=" * 60)
    print(f"📊 Total samples extracted: {len(all_data)}")

    if not all_data:
        print("❌ No data extracted. Exiting.")
        return

    # Create DataFrame
    df = pd.DataFrame(all_data)

    # Merge with ground truth
    print(f"\n🔗 Merging with ground truth...")
    df['ground_truth_answer'] = df['question_id'].map(ground_truth_map)

    # Check for missing ground truth
    missing_gt = df['ground_truth_answer'].isna().sum()
    if missing_gt > 0:
        print(f"⚠️  Warning: {missing_gt} samples have missing ground truth")
        print(f"   Missing question_ids: {df[df['ground_truth_answer'].isna()]['question_id'].head(5).tolist()}")
    else:
        print(f"   ✓ All samples have ground truth answers")

    # Reorder columns to match other datasets
    df = df[['question_id', 'image_id', 'question', 'ground_truth_answer', 'model_answer']]

    # Sort by question_id
    df = df.sort_values('question_id')

    # Save to CSV
    df.to_csv(output_csv, index=False)

    print(f"💾 Saved to: {output_csv}")
    print()

    # Print summary statistics
    print("📈 Dataset Summary:")
    print(f"   Total samples: {len(df)}")
    print(f"   Unique images: {df['image_id'].nunique()}")
    print(f"   Unique questions: {df['question_id'].nunique()}")
    print()

    # Show sample
    print("🔍 Sample rows:")
    print(df.head(3).to_string(max_colwidth=50))
    print()

    # Check for missing data
    missing = df.isnull().sum()
    if missing.any():
        print("⚠️  Missing data:")
        print(missing[missing > 0])
    else:
        print("✅ No missing data")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Build hallucination dataset from SmolVLM HDF5 files")
    parser.add_argument('--input-dir', type=str, default='./smol_output',
                       help='Directory containing .h5 files')
    parser.add_argument('--output-csv', type=str, default='./smolvlm_hallucination_dataset.csv',
                       help='Output CSV file path')
    parser.add_argument('--reference-csv', type=str,
                       default='../Gemma_3/gemma3_hallucination_dataset.csv',
                       help='Reference CSV with ground truth answers')

    args = parser.parse_args()

    # Convert to absolute paths
    input_dir = os.path.abspath(args.input_dir)
    output_csv = os.path.abspath(args.output_csv)
    reference_csv = os.path.abspath(args.reference_csv)

    if not os.path.exists(input_dir):
        print(f"❌ Input directory does not exist: {input_dir}")
        return

    if not os.path.exists(reference_csv):
        print(f"❌ Reference CSV does not exist: {reference_csv}")
        print(f"   Please provide a valid reference CSV with ground truth")
        return

    build_hallucination_dataset(input_dir, output_csv, reference_csv)
    print(f"\n✅ Done! Dataset saved to: {output_csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build Hallucination Dataset CSV from LLama-3.2-11B-Vision HDF5 Files

Extracts: question_id, image_id, question, ground_truth_answer, model_answer
from all .h5 files and creates a CSV for hallucination analysis.
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
            for question_id in f.keys():
                try:
                    q_group = f[question_id]

                    # Extract text fields
                    question = q_group['question'][()].decode('utf-8') if 'question' in q_group else ''
                    image_id = q_group['image_id'][()].decode('utf-8') if 'image_id' in q_group else ''
                    ground_truth = q_group['ground_truth_answer'][()].decode('utf-8') if 'ground_truth_answer' in q_group else ''
                    model_answer = q_group['answer'][()].decode('utf-8') if 'answer' in q_group else ''

                    data_rows.append({
                        'question_id': question_id,
                        'image_id': image_id,
                        'question': question,
                        'ground_truth_answer': ground_truth,
                        'model_answer': model_answer
                    })

                except Exception as e:
                    print(f"  Warning: Failed to extract {question_id}: {e}")
                    continue

    except Exception as e:
        print(f"Error reading {h5_path}: {e}")
        return []

    return data_rows


def build_hallucination_dataset(input_dir, output_csv):
    """Build hallucination dataset from all .h5 files in directory"""

    print("=" * 60)
    print("Building Hallucination Dataset for LLama-3.2-11B-Vision")
    print("=" * 60)

    # Find all .h5 files
    h5_pattern = os.path.join(input_dir, "*.h5")
    h5_files = sorted(glob.glob(h5_pattern))

    if not h5_files:
        print(f"❌ No .h5 files found in {input_dir}")
        return

    print(f"📁 Found {len(h5_files)} HDF5 files")
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
    parser = argparse.ArgumentParser(description="Build hallucination dataset from LLama-3.2-11B-Vision HDF5 files")
    parser.add_argument('--input-dir', type=str, default='./llama_output',
                       help='Directory containing .h5 files')
    parser.add_argument('--output-csv', type=str, default='./llama32_hallucination_dataset.csv',
                       help='Output CSV file path')

    args = parser.parse_args()

    # Convert to absolute paths
    input_dir = os.path.abspath(args.input_dir)
    output_csv = os.path.abspath(args.output_csv)

    if not os.path.exists(input_dir):
        print(f"❌ Input directory does not exist: {input_dir}")
        return

    build_hallucination_dataset(input_dir, output_csv)
    print(f"\n✅ Done! Dataset saved to: {output_csv}")


if __name__ == "__main__":
    main()

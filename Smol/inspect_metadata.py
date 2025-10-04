#!/usr/bin/env python3
"""Check exact structure of metadata in H5 files"""

import h5py
from pathlib import Path

h5_file = Path("/root/akhil/HALP_EACL_Models/Models/Smol_VL/smolvlm_output/smolvlm_2.2b_embeddings_part_001.h5")

with h5py.File(h5_file, 'r') as f:
    sample = f['question_comb_1']

    print("="*80)
    print("DETAILED STRUCTURE INSPECTION")
    print("="*80)

    print(f"\nAll keys in sample: {list(sample.keys())}")
    print(f"\nAll attributes in sample: {list(sample.attrs.keys())}")

    print("\n" + "="*80)
    print("CHECKING EACH KEY TYPE")
    print("="*80)

    for key in sample.keys():
        item = sample[key]
        if isinstance(item, h5py.Group):
            print(f"\n'{key}': GROUP")
            print(f"  Sub-keys: {list(item.keys())}")
        elif isinstance(item, h5py.Dataset):
            print(f"\n'{key}': DATASET")
            print(f"  Shape: {item.shape}")
            print(f"  Dtype: {item.dtype}")
            if item.shape == ():  # Scalar dataset
                print(f"  Value: {item[()]}")
            else:
                print(f"  Value preview: {item[:5] if len(item) > 5 else item[:]}")

    print("\n" + "="*80)
    print("CHECKING ATTRIBUTES")
    print("="*80)

    for attr_name in sample.attrs.keys():
        print(f"\n'{attr_name}': {sample.attrs[attr_name]}")

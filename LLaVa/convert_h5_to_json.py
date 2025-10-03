import h5py
import json
import numpy as np
import os
import argparse

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, bytes):
        return obj.decode('utf-8')
    return obj

def convert_h5_to_json(h5_file, json_file):
    """Convert a single H5 file to JSON format"""
    print(f"Converting {h5_file} to {json_file}...")
    
    data = {}
    
    with h5py.File(h5_file, 'r') as f:
        for sample_id in f.keys():
            sample_group = f[sample_id]
            sample_data = {}
            
            for key in sample_group.keys():
                item = sample_group[key]
                
                if isinstance(item, h5py.Group):
                    # Handle nested groups (vision_token_representation, query_token_representation)
                    group_data = {}
                    for layer_name in item.keys():
                        layer_data = item[layer_name][()]
                        group_data[layer_name] = convert_numpy_types(layer_data)
                    sample_data[key] = group_data
                else:
                    # Handle datasets
                    value = item[()]
                    sample_data[key] = convert_numpy_types(value)
            
            data[sample_id] = sample_data
    
    # Write to JSON
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Show file sizes
    h5_size = os.path.getsize(h5_file) / (1024 * 1024)
    json_size = os.path.getsize(json_file) / (1024 * 1024)
    
    print(f"  ✅ Conversion complete!")
    print(f"     H5 size: {h5_size:.2f} MB")
    print(f"     JSON size: {json_size:.2f} MB")
    print(f"     Size ratio: {json_size/h5_size:.2f}x larger")

def main():
    parser = argparse.ArgumentParser(description='Convert LLaVa H5 files to JSON')
    parser.add_argument('--input', type=str, help='Input H5 file (optional, will convert all if not specified)')
    parser.add_argument('--input-dir', type=str, default='./llava_output', help='Input directory containing H5 files')
    parser.add_argument('--output-dir', type=str, default='./llava_output', help='Output directory for JSON files')
    args = parser.parse_args()
    
    if args.input:
        # Convert single file
        h5_file = args.input
        json_file = h5_file.replace('.h5', '.json')
        convert_h5_to_json(h5_file, json_file)
    else:
        # Convert all H5 files in input directory
        import glob
        h5_files = sorted(glob.glob(os.path.join(args.input_dir, '*.h5')))
        
        if not h5_files:
            print(f"No H5 files found in {args.input_dir}")
            return
        
        print(f"Found {len(h5_files)} H5 files to convert")
        print("="*60)
        
        for h5_file in h5_files:
            basename = os.path.basename(h5_file)
            json_file = os.path.join(args.output_dir, basename.replace('.h5', '.json'))
            convert_h5_to_json(h5_file, json_file)
            print()
        
        print("="*60)
        print(f"✅ All conversions complete! Converted {len(h5_files)} files.")

if __name__ == "__main__":
    main()

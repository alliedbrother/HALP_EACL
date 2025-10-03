# Llama 3.2 Vision Extraction - Background Processing Guide

## Features ✨

✅ **Per-image logging** - Shows success/failure for every single image
✅ **Empty gt_answer handling** - Works even if ground truth answers are missing
✅ **Background execution** - Continues running even if you disconnect
✅ **Progress tracking** - Real-time ETA, speed, and GPU memory monitoring
✅ **Checkpointing** - Saves every 1000 images to HDF5 files
✅ **Full dataset processing** - Processes all images in your dataset

## Quick Start 🚀

### 1. Start Extraction (Background Mode)

```bash
cd /root/halp/scripts

./run_llama_extraction_background.sh \
    /path/to/vqa_dataset.csv \
    /path/to/images_directory \
    ./output_directory \
    meta-llama/Llama-3.2-11B-Vision-Instruct \
    ./model_cache \
    1000
```

**Parameters:**
- `$1`: Path to VQA CSV file (required)
- `$2`: Directory containing images (required)
- `$3`: Output directory (default: `./llama_output`)
- `$4`: Model name (default: `meta-llama/Llama-3.2-11B-Vision-Instruct`)
- `$5`: Model cache directory (default: `./model_cache`)
- `$6`: Checkpoint interval (default: `1000`)

### 2. Monitor Progress

```bash
# View real-time logs
./manage_llama_extraction.sh ./output_directory logs

# Check status
./manage_llama_extraction.sh ./output_directory status

# View statistics
./manage_llama_extraction.sh ./output_directory stats

# List checkpoints
./manage_llama_extraction.sh ./output_directory checkpoints
```

### 3. Stop Extraction

```bash
./manage_llama_extraction.sh ./output_directory stop
```

## Example Usage 📝

```bash
# Start extraction
./run_llama_extraction_background.sh \
    /data/vqav2_val.csv \
    /data/coco_images \
    /data/llama_embeddings

# In another terminal or after reconnecting
cd /root/halp/scripts

# Check if still running
./manage_llama_extraction.sh /data/llama_embeddings status

# View latest stats
./manage_llama_extraction.sh /data/llama_embeddings stats

# Watch logs in real-time
./manage_llama_extraction.sh /data/llama_embeddings logs
```

## Log Output Example 📊

```
🔄 [12345] Processing image: COCO_val2014_000000123456.jpg
✅ [12345] Successfully extracted embeddings | Total: 543/5000

🔄 [12346] Processing image: COCO_val2014_000000123457.jpg
❌ [12346] Failed to process: Image corrupted

📊 Progress: 600/5000 | Speed: 2.34 samples/sec | ETA: 0 days 00:31:24 | Failed: 12

💾 GPU Memory: Allocated=8.34GB, Reserved=10.12GB

💾 Saved checkpoint at 1000 samples
```

## Output Files 📁

```
output_directory/
├── llama_extraction.log                          # Detailed extraction log
├── llama_extraction_background.log               # Background process log
├── llama_extraction.pid                          # Process ID
├── llama3_2_11b_vision_embeddings_part_001.h5   # First 1000 samples
├── llama3_2_11b_vision_embeddings_part_002.h5   # Next 1000 samples
└── ...
```

## HDF5 File Structure 🗂️

Each checkpoint file contains:

```
llama3_2_11b_vision_embeddings_part_001.h5
├── [question_id_1]/
│   ├── question (string)
│   ├── image_id (string)
│   ├── ground_truth_answer (string, empty if missing)
│   ├── answer (string, generated)
│   ├── vision_only_representation [1280] (float32)
│   ├── vision_token_representation/
│   │   ├── layer_0 [4096]
│   │   ├── layer_10 [4096]
│   │   ├── layer_20 [4096]
│   │   ├── layer_30 [4096]
│   │   └── layer_39 [4096]
│   └── query_token_representation/
│       ├── layer_0 [4096]
│       ├── layer_10 [4096]
│       ├── layer_20 [4096]
│       ├── layer_30 [4096]
│       └── layer_39 [4096]
```

## Troubleshooting 🔧

### Process won't start
```bash
# Check if already running
./manage_llama_extraction.sh ./output_directory status

# If stuck, force stop and restart
./manage_llama_extraction.sh ./output_directory stop
```

### Out of memory
```bash
# Reduce checkpoint interval (processes fewer images before saving)
./run_llama_extraction_background.sh dataset.csv images/ output/ model cache 500
```

### View errors only
```bash
grep "❌" output_directory/llama_extraction.log
```

### Find last processed image
```bash
tail -20 output_directory/llama_extraction.log | grep "✅"
```

## Advanced: Screen Session (Alternative Method)

If you prefer using `screen`:

```bash
# Start screen session
screen -S llama_extraction

# Run extraction (foreground)
python3 /root/halp/scripts/llama_extract.py \
    --vqa-dataset /path/to/dataset.csv \
    --images-dir /path/to/images \
    --output-dir /path/to/output \
    --checkpoint-interval 1000

# Detach: Press Ctrl+A then D

# Reattach later
screen -r llama_extraction

# List sessions
screen -ls
```

## Performance Tips ⚡

- **Checkpoint interval**: 1000 is optimal for 24GB VRAM
- **Expected speed**: ~2-3 samples/sec on RTX 4090
- **Disk space**: ~170KB per sample (uncompressed), ~50KB with gzip
- **GPU memory**: ~10-12GB for 11B model with bfloat16

## Support 💬

For issues, check:
1. Log files in output directory
2. GPU memory with `nvidia-smi`
3. Disk space with `df -h`

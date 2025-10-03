# Phi-4 Multimodal VQA Embedding Extractor

This script extracts embeddings from Microsoft's Phi-4 Multimodal model for Visual Question Answering (VQA) tasks.

## Model Information

- **Model**: microsoft/Phi-4-multimodal-instruct
- **Size**: 5.6B parameters (3.8B language model + 370M vision LoRA + adapters)
- **Vision Encoder**: SigLIP-400M
- **Language Model**: Phi-4-Mini-Instruct (32 layers)
- **Hidden Size**: 3,072
- **Context Length**: 128K tokens
- **Image Resolution**: 448×448 (dynamic multi-crop supported)

## Requirements

### Dependencies

```bash
pip install torch==2.6.0
pip install transformers==4.48.2
pip install accelerate==1.3.0
pip install pillow==11.1.0
pip install torchvision==0.21.0
pip install peft==0.13.2
pip install flash-attn==2.7.4.post1
pip install h5py pandas numpy tqdm
```

**Note**: Flash Attention 2 requires:
- CUDA 11.8 or higher
- GPU with compute capability 8.0+ (A100, H100, RTX 4090, etc.)
- For older GPUs (V100), the script will fall back to eager attention

### Hardware Requirements

- **Minimum**: 16GB VRAM (with float16)
- **Recommended**: 24GB+ VRAM (A6000, A100, RTX 4090)
- **CPU RAM**: 32GB+

## Usage

### Test Mode (3 samples)

```bash
cd scripts/phi4

python phi4_extract.py \
  --vqa-dataset ../../data/_tiny_from_images.csv \
  --images-dir ../../images \
  --output-dir ../../outputs/phi4 \
  --test
```

### Full Dataset Processing

```bash
python phi4_extract.py \
  --vqa-dataset ../../data/sampled_10k_relational_dataset.csv \
  --images-dir ../../images \
  --output-dir ../../outputs/phi4 \
  --checkpoint-interval 1000
```

### Command-Line Arguments

- `--vqa-dataset`: Path to VQA CSV dataset (required)
- `--images-dir`: Directory containing images (required)
- `--output-dir`: Output directory for embeddings (default: ./output)
- `--model`: Model path (default: microsoft/Phi-4-multimodal-instruct)
- `--cache-dir`: Model cache directory (default: ./model_cache)
- `--checkpoint-interval`: Save every N samples (default: 1000)
- `--test`: Test mode - process only 3 samples

## Extracted Representations

The script extracts 4 types of representations:

### 1. Vision-Only Representation
- **Source**: SigLIP vision encoder (before language fusion)
- **Shape**: [1152] (pooled across spatial dimensions)
- **Description**: Pure visual features without language context

### 2. Vision Token Representation
- **Source**: Language decoder hidden states at `<|image_1|>` token position
- **Layers**: [0, 8, 16, 24, 31] (0%, 25%, 50%, 75%, 100% of 32 layers)
- **Shape**: [3072] per layer
- **Description**: Multimodal representation at vision-language fusion point

### 3. Query Token Representation
- **Source**: Language decoder hidden states at last token position
- **Layers**: [0, 8, 16, 24, 31]
- **Shape**: [3072] per layer
- **Description**: Final reasoning state after processing question and image

### 4. Generated Answer
- **Source**: Model generation (greedy decoding, max 200 tokens)
- **Type**: String
- **Description**: Model's answer to the VQA question

## Output Format

### HDF5 File Structure

```
phi4_multimodal_embeddings_part_001.h5
├── [attributes]
│   ├── model_name: "microsoft/Phi-4-multimodal-instruct"
│   ├── model_type: "phi-4-multimodal"
│   ├── device: "cuda"
│   ├── dtype: "float16"
│   ├── created_at: "2025-10-02T..."
│   └── num_samples: 1000
│
├── question_comb_957/
│   ├── image_id: "AMBER_1.jpg"
│   ├── question: "Is there direct contact..."
│   ├── ground_truth_answer: "n, o"
│   ├── vision_only_representation: [1152] array
│   ├── vision_token_representation/
│   │   ├── layer_0: [3072] array
│   │   ├── layer_8: [3072] array
│   │   ├── layer_16: [3072] array
│   │   ├── layer_24: [3072] array
│   │   └── layer_31: [3072] array
│   ├── query_token_representation/
│   │   └── ... (same structure)
│   └── answer: "Yes, there is direct contact..."
│
└── ... (more questions)
```

## Architecture Details

### Phi-4 Multimodal Architecture

```
┌─────────────────────────────────────────┐
│   Phi-4 Multimodal (5.6B) Architecture  │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────────────────────────────┐  │
│  │  SigLIP Vision Encoder (400M)    │  │ ← Process images
│  │  - 27 layers, hidden_size=1152   │  │
│  │  - Output: [B, 1024, 1152]       │  │
│  └──────────────┬───────────────────┘  │
│                 │                       │
│                 ▼                       │
│  ┌──────────────────────────────────┐  │
│  │  Vision Projector (2-layer MLP)  │  │ ← Map to text space
│  │  - 1152 → 3072 dimensions        │  │
│  └──────────────┬───────────────────┘  │
│                 │                       │
│                 ▼                       │
│  ┌──────────────────────────────────┐  │
│  │  Phi-4-Mini Language Decoder     │  │ ← Process text + vision
│  │  - 32 layers, hidden_size=3072   │  │
│  │  - Frozen, with Vision LoRA      │  │
│  │  - Input: text + vision tokens   │  │
│  └──────────────────────────────────┘  │
│                                         │
└─────────────────────────────────────────┘
```

### Key Features

1. **Mixture-of-LoRAs**: Uses LoRA adapters instead of fully fine-tuned cross-attention
2. **Frozen Language Model**: Base Phi-4-Mini remains frozen during multimodal training
3. **Dynamic Multi-Crop**: Adapts to different image resolutions
4. **Token Compression**: Optional pooling to reduce vision token count
5. **Modality-Specific Routing**: Separate adapters for vision and audio

### Prompt Format

Phi-4 uses a specific chat template:

```
<|user|><|image_1|>What is shown in this image?<|end|><|assistant|>
```

Special tokens:
- `<|user|>`: User message prefix
- `<|image_1|>`: Image placeholder (token ID: 200010)
- `<|end|>`: End of turn marker
- `<|assistant|>`: Assistant response prefix

## Troubleshooting

### Flash Attention Issues

If you get errors related to Flash Attention:

1. **For V100 or older GPUs**: The script automatically falls back to eager attention
2. **For A100/H100**: Ensure flash-attn is properly installed:
   ```bash
   pip install flash-attn --no-build-isolation
   ```

### Out of Memory (OOM)

1. Reduce batch size (already set to 1)
2. Reduce `checkpoint_interval` to save more frequently
3. Enable gradient checkpointing (not needed for inference)
4. Use smaller image resolution (requires processor modification)

### LoRA Adapter Not Found

If vision adapter fails to load:
```
RuntimeError: Vision adapter not found
```

Solution:
1. Check internet connection (adapter downloads from HuggingFace)
2. Verify cache directory has write permissions
3. Check HuggingFace token if model requires authentication

### Model Performance

Expected processing speed:
- **A100 (40GB)**: ~2-3 samples/second
- **RTX 4090**: ~1.5-2 samples/second
- **A6000**: ~1-1.5 samples/second

## Comparison with Llama 3.2 Vision

| Feature | Phi-4 Multimodal | Llama 3.2 Vision 11B |
|---------|------------------|----------------------|
| **Parameters** | 5.6B | 11B |
| **Layers** | 32 | 40 |
| **Hidden Size** | 3072 | 4096 |
| **Vision Encoder** | SigLIP-400M | MllamaVision |
| **Vision Adapter** | LoRA (370M) | Cross-attention |
| **Image Token** | `<|image_1|>` (200010) | `<|image|>` (128256) |
| **Context Length** | 128K | 128K |
| **VRAM (fp16)** | ~16GB | ~24GB |

## References

- **Model Card**: https://huggingface.co/microsoft/Phi-4-multimodal-instruct
- **Technical Report**: https://arxiv.org/abs/2503.01743
- **Documentation**: https://huggingface.co/docs/transformers/model_doc/phi4_multimodal
- **GitHub Cookbook**: https://github.com/microsoft/PhiCookBook

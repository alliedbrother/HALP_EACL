# Phi-4 Multimodal Embedding Extraction - Detailed Architecture Explanation

## Overview

The script extracts **4 types of representations** from the Phi-4 Multimodal (5.6B) model:
1. **Vision-only representation** (from SigLIP vision encoder, before cross-attention)
2. **Vision token representation** (from language decoder at image token position)
3. **Query token representation** (from language decoder at end of prompt)
4. **Generated answer** (text output)

---

## Phi-4 Multimodal Architecture

```
                    Phi-4 Multimodal (5.6B) Architecture
                    ====================================

Input: Image + Text Question

┌─────────────────────────────────────────────────────────────────┐
│                         IMAGE PATH                              │
│                                                                 │
│  Image (PIL)                                                    │
│      ↓                                                          │
│  Processor → pixel_values [B, C, H, W]                         │
│            → image patches                                      │
│      ↓                                                          │
│  ┌──────────────────────────────────────┐                      │
│  │   Vision Encoder (SigLIP ViT)        │                      │
│  │   - SigLIP (Google's vision model)   │                      │
│  │   - Processes image patches          │                      │
│  │   - Output: vision features          │                      │
│  └──────────────────────────────────────┘                      │
│      ↓                                                          │
│  Vision Features                                                │
│  (From img_processor in model.model.embed_tokens_extend)       │
│      ↓                                                          │
│  **EXTRACTION POINT 1: vision_only_representation**            │
│  - Pooled (averaged) vision features → [448]                  │
│  - This is BEFORE projection to language space                 │
│  - Pure vision features without language influence             │
│      ↓                                                          │
│  Vision Projection Layer                                       │
│  (Projects vision features to language embedding space)        │
│      ↓                                                          │
│  Projected Vision Embeddings [3072-dim]                        │
│  (Ready for language model)                                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         TEXT PATH                               │
│                                                                 │
│  Question: "Is the forest lit by a full moon?"                  │
│      ↓                                                          │
│  Phi-4 Template:                                                │
│  "<|user|><|image_1|>Is the forest lit by a full moon?<|end|>  │
│   <|assistant|>"                                                │
│      ↓                                                          │
│  Tokenizer → input_ids                                          │
│  [29871, 29989, 1792, 29989, 200010, 1317, 278, ...]          │
│                              ^^^^^^                             │
│                              200010 = <|image_1|> token         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    LANGUAGE DECODER                             │
│                    (32 Transformer Layers)                      │
│                                                                 │
│  Input Sequence:                                                │
│  [<|user|>] + [<|image_1|>] + [question_tokens] + [<|end|>]   │
│  + [<|assistant|>]                                             │
│      ↓                                                          │
│  Token Embedding Layer (model.model.embed_tokens_extend)       │
│  - Regular tokens → embeddings [3072-dim]                      │
│  - <|image_1|> → replaced with projected vision features       │
│      ↓                                                          │
│  ┌─────────────────────────────────────────┐                   │
│  │  Layer 0 (Transformer Block)             │                  │
│  │  - Self-Attention                        │                  │
│  │  - Feed-Forward Network                  │                  │
│  │  Output: hidden_states[1] [seq, 3072]   │ ← EXTRACTION 2   │
│  └─────────────────────────────────────────┘                   │
│      ↓                                                          │
│  ┌─────────────────────────────────────────┐                   │
│  │  Layer 8 (25% through)                   │                  │
│  │  Output: hidden_states[9] [seq, 3072]   │ ← EXTRACTION 2   │
│  └─────────────────────────────────────────┘                   │
│      ↓                                                          │
│  ┌─────────────────────────────────────────┐                   │
│  │  Layer 16 (50% through)                  │                  │
│  │  Output: hidden_states[17] [seq, 3072]  │ ← EXTRACTION 2   │
│  └─────────────────────────────────────────┘                   │
│      ↓                                                          │
│  ┌─────────────────────────────────────────┐                   │
│  │  Layer 24 (75% through)                  │                  │
│  │  Output: hidden_states[25] [seq, 3072]  │ ← EXTRACTION 2   │
│  └─────────────────────────────────────────┘                   │
│      ↓                                                          │
│  ┌─────────────────────────────────────────┐                   │
│  │  Layer 31 (Final layer)                  │                  │
│  │  Output: hidden_states[32] [seq, 3072]  │ ← EXTRACTION 2   │
│  └─────────────────────────────────────────┘                   │
│      ↓                                                          │
│  **EXTRACTION POINT 2 & 3:**                                    │
│  For each selected layer (0, 8, 16, 24, 31):                  │
│    - Extract hidden state at <|image_1|> position              │
│      → vision_token_representation[layer_k] [3072]             │
│    - Extract hidden state at last prompt token                 │
│      → query_token_representation[layer_k] [3072]              │
│      ↓                                                          │
│  LM Head (Linear projection to vocab)                          │
│      ↓                                                          │
│  Logits → Sampling → Generated Tokens                          │
│      ↓                                                          │
│  **EXTRACTION POINT 4: answer (text)**                         │
│  Decoded text: "No, the forest is not lit by a full moon..."   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detailed Extraction Points

### 1. Vision-Only Representation (`_extract_vision_representation`, lines 166-195)

**What:** Pure visual features from the SigLIP vision encoder, BEFORE any language interaction.

**Architecture Location:**
- Module: `model.model.embed_tokens_extend.image_embed.img_processor` (SigLIP ViT)
- This is the vision encoder component within Phi-4's embedding layer
- Output: Vision features (exact format depends on processor implementation)

**Extraction Process:**
```python
# Line 174: Use image processor to get vision features
vis_inputs = self.processor.image_processor(images=[image], return_tensors="pt")

# Line 177-180: Extract pre-computed embeddings
if 'input_image_embeds' in vis_inputs:
    feats = vis_inputs['input_image_embeds'].to(self.model.device)

    # Line 183-184: Pool over all spatial/tile dimensions
    while feats.dim() > 2:
        feats = feats.mean(dim=1)

    # Line 187: Final pooled vector [448]
    rep = feats.squeeze(0).to(torch.float32).cpu().numpy()
```

**Why this matters:**
- Represents what the model "sees" in the image
- No language contamination yet
- Useful for probing pure visual understanding
- 448 dimensions = SigLIP hidden size (smaller than Llama's 7680)

**Flow:**
```
Image → Image Processor → SigLIP ViT →
Vision Features (variable dims) → Average pool → [448]
```

**Important Note:**
Unlike Llama 3.2, Phi-4's vision encoder doesn't have a direct forward pass method that returns hidden states. Instead, the processor pre-computes embeddings (`input_image_embeds`), which we extract and pool.

---

### 2. Vision Token Representation (`_extract_decoder_embeddings`, lines 197-236)

**What:** Hidden states from the language decoder AT the position of the `<|image_1|>` token placeholder.

**Architecture Location:**
- Module: `model` (full Phi4MMModel)
- Layers: 32 transformer layers in the language decoder
- Selected layers: `[0, 8, 16, 24, 31]` (0%, 25%, 50%, 75%, 100% through decoder)

**Extraction Process:**

**Step 1: Build Phi-4 formatted prompt (lines 125-132)**
```python
# Line 126: Phi-4 uses specific template format
prompt = f'<|user|><|image_1|>{question}<|end|><|assistant|>'
# Example: "<|user|><|image_1|>Is the forest lit by a full moon?<|end|><|assistant|>"

# Line 128-132: Process into model inputs
inputs = self.processor(
    text=prompt,
    images=[image],
    return_tensors="pt"
).to(self.model.device)
```

**Step 2: Find the image token position (lines 210-215)**
```python
# Line 95: Image token ID is hard-coded for Phi-4
self.image_token_id = 200010  # <|image_1|> in Phi-4

# Line 211: Find where <|image_1|> token is in the sequence
image_positions = (input_ids == self.image_token_id).nonzero()
# self.image_token_id = 200010 (the special <|image_1|> token)

# Line 213: Use the LAST occurrence (in case of multiple images)
vision_token_boundary = int(image_positions[-1].item())
```

Example input_ids sequence for Phi-4:
```
[29871, 29989, 1792, 29989, 200010, 1317, 278, 13569, ...]
                            ^^^^^^
                            <|image_1|> token at some position
                            This is vision_token_boundary
```

**Step 3: Forward pass to get hidden states (lines 199-200)**
```python
# Line 199-200: Single forward pass with all hidden states
with torch.no_grad():
    outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)

# outputs.hidden_states = tuple of 33 tensors:
#   [0]: embeddings [batch, seq, 3072]
#   [1]: layer_0 output [batch, seq, 3072]
#   [2]: layer_1 output [batch, seq, 3072]
#   ...
#   [32]: layer_31 output [batch, seq, 3072]
```

**Step 4: Extract hidden states at image position (lines 228-234)**
```python
for k in target_layers:  # [0, 8, 16, 24, 31]
    # Line 230: Get hidden state from layer k
    # +1 offset because hidden_states[0] = input embeddings
    layer_h = hidden_states[k + 1][0]  # [seq_len, 3072]

    # Line 231: Extract the vector at image token position
    vt_vec = layer_h[vision_token_boundary]  # [3072]

    # Store as numpy array
    vision_token_reps[f"layer_{k}"] = vt_vec.cpu().numpy()
```

**Why this matters:**
- Shows how the model represents the image AFTER being processed by the language model
- Captures multimodal fusion at different depths
- Different layers show different levels of abstraction:
  - Layer 0: Early embedding with vision features injected
  - Layer 8: Early semantic processing
  - Layer 16: Mid-level multimodal reasoning
  - Layer 24: High-level task-specific features
  - Layer 31: Final representation before generation

**Flow:**
```
Input tokens → Embedding (with vision injection at <|image_1|>) →
Layer 0 → ... → Layer k → ...
    ↓              ↓
    At image token position: vision_token_rep[layer_0] [3072]
                            vision_token_rep[layer_k] [3072]
```

---

### 3. Query Token Representation (`_extract_decoder_embeddings`, lines 197-236)

**What:** Hidden states from the language decoder AT the END of the input prompt (last non-padded token).

**Architecture Location:**
- Same as vision token representation
- Different extraction position in the sequence

**Extraction Process:**

**Step 1: Find the last prompt token position (lines 217-222)**
```python
# Line 218-221: Find last non-padding token
if attn_mask is not None:
    last_idx = int(attn_mask[0].sum().item()) - 1
else:
    last_idx = int(inputs["input_ids"].shape[1] - 1)
query_token_boundary = max(0, last_idx)
```

Example sequence for Phi-4:
```
Input: "<|user|><|image_1|>Is the forest lit?<|end|><|assistant|>"
Tokens: [29871, 29989, 1792, 29989, 200010, ..., 29989, 465, 29989]
                                                              ^^^^^
                                                              Last token position
                                                              This is query_token_boundary
```

**Step 2: Extract hidden states at this position (lines 228-234)**
```python
for k in target_layers:  # [0, 8, 16, 24, 31]
    layer_h = hidden_states[k + 1][0]  # [seq_len, 3072]

    # Line 232: Extract the vector at last prompt token position
    qt_vec = layer_h[query_token_boundary]  # [3072]

    query_token_reps[f"layer_{k}"] = qt_vec.cpu().numpy()
```

**Why this matters:**
- Represents the model's understanding of the QUESTION + IMAGE together
- This is the "query" state that the model will use to generate the answer
- Contains both visual and textual information integrated
- Shows the multimodal understanding at different depths

**Flow:**
```
Input tokens → Embedding → Layer 0 → ... → Layer k → ...
                              ↓              ↓
                              At last position: query_token_rep[layer_0] [3072]
                                               query_token_rep[layer_k] [3072]
```

---

### 4. Generated Answer (`_generate_answer`, lines 150-164)

**What:** The actual text response generated by the model.

**Architecture Location:**
- Full model with autoregressive generation
- Uses all layers iteratively

**Extraction Process:**
```python
# Line 152-157: Generate tokens with float16 precision
with torch.autocast(device_type="cuda", dtype=torch.float16):
    outputs = self.model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False  # Greedy decoding (argmax)
    )

# Line 159: Skip the prompt tokens (only keep generated part)
gen_tokens = outputs[0, inputs["input_ids"].size(1):]

# Line 160: Decode tokens to text
text = self.processor.decode(gen_tokens, skip_special_tokens=True)
```

**Why this matters:**
- The model's actual prediction/answer to the VQA task
- Can be compared with ground truth to evaluate performance
- Shows the final output of the multimodal reasoning

---

## Layer Selection Strategy (lines 111-120)

```python
def _target_layers(self) -> List[int]:
    # 0, n/4, n/2, 3n/4, n-1
    nl = self.num_layers  # 32 for Phi-4
    return sorted(set([
        0,           # Layer 0  (0% through)
        nl // 4,     # Layer 8  (25% through)
        nl // 2,     # Layer 16 (50% through)
        (3 * nl) // 4,  # Layer 24 (75% through)
        nl - 1       # Layer 31 (100% through)
    ]))
```

**Why these specific layers?**
- Captures progression of information processing
- Layer 0: Initial multimodal fusion
- Layer 8: Early semantic processing
- Layer 16: Mid-level reasoning
- Layer 24: High-level abstraction
- Layer 31: Final pre-generation state

---

## Key Technical Details

### Hidden States Structure

When you call:
```python
outputs = model(**inputs, output_hidden_states=True)
```

You get:
```python
outputs.hidden_states = (
    embeddings,      # [batch, seq, 3072] - index 0
    layer_0_output,  # [batch, seq, 3072] - index 1
    layer_1_output,  # [batch, seq, 3072] - index 2
    ...
    layer_31_output  # [batch, seq, 3072] - index 32
)
# Total: 33 tensors (embeddings + 32 layers)
```

**Important:** That's why line 230 uses `hidden_states[k + 1]` - the +1 offset accounts for the embedding layer at index 0.

### Phi-4 Template Format

Phi-4 uses a specific chat template:
```
<|user|><|image_1|>{question}<|end|><|assistant|>
```

Special tokens:
- `<|user|>`: Marks user turn
- `<|image_1|>`: Placeholder for first image (ID: 200010)
- `<|end|>`: End of user turn
- `<|assistant|>`: Start of assistant turn

### Vision Feature Dimensions

- **Vision encoder output:** 448 dimensions
  - This is from SigLIP ViT
  - Much smaller than Llama's 7680

- **Language decoder hidden states:** 3072 dimensions
  - This is the native hidden size of Phi-4 language model
  - Smaller than Llama's 4096

### Token Sequence Example

For input: `"<|user|><|image_1|>Is the forest lit?<|end|><|assistant|>"`

Tokenized (example):
```
Position:  0     1      2     3      4      5      6    ...   N
Token:    [user][delim][img1][Is]  [the] [forest][lit] ... [asst]
ID:       29871  29989  200010 1317   278   13569  ... 465   29989
                        ^                                      ^
                vision_token_boundary              query_token_boundary
```

The exact positions vary based on tokenization, but the script finds them dynamically.

---

## Phi-4 vs Llama 3.2 Comparison

| Feature | Llama 3.2 11B | Phi-4 5.6B |
|---------|---------------|------------|
| **Vision Encoder** | MllamaVisionModel | SigLIP ViT |
| **Vision dims** | 7680 | 448 |
| **Language layers** | 40 | 32 |
| **Language hidden** | 4096 | 3072 |
| **Image token** | `<|image|>` (128256) | `<|image_1|>` (200010) |
| **Template** | Chat template | `<|user|><|image_1|>...<|end|><|assistant|>` |
| **Vision access** | Direct `vision_model` forward | Pre-computed via processor |
| **Model size** | 11B params | 5.6B params |

---

## Summary Table

| Representation Type | Source Module | Extraction Point | Shape | Purpose |
|---------------------|---------------|------------------|-------|---------|
| `vision_only_representation` | SigLIP ViT | Pooled vision features | [448] | Pure visual features before language |
| `vision_token_representation` | Language Decoder | Hidden state at `<|image_1|>` position, layers [0,8,16,24,31] | 5 × [3072] | Image representation in language space |
| `query_token_representation` | Language Decoder | Hidden state at last prompt token, layers [0,8,16,24,31] | 5 × [3072] | Question+Image integrated representation |
| `answer` | Full Model | Generated text output | String | Model's VQA answer |

---

## Visual Flow Summary

```
                  ┌─────────────┐
                  │   Image     │
                  └──────┬──────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  SigLIP ViT          │
              │  (Vision Encoder)    │
              └──────┬───────────────┘
                     │
                     ├──► EXTRACT 1: vision_only_rep [448]
                     │    (pooled vision features)
                     │
                     ▼
              ┌──────────────────┐
              │  Vision Projector│
              └──────┬───────────┘
                     │
                     ▼
              ┌─────────────────────────────┐
              │  Language Decoder           │
              │  (32 transformer layers)    │
              │                             │
    Text ────►│  [user][IMG1][tok][end][asst]│
              │          ▲           ▲      │
              │          │           │      │
              │          │           └─► EXTRACT 3: query_token_rep
              │          │              (at last position, layers 0,8,16,24,31)
              │          │              5 × [3072]
              │          │
              │          └─────► EXTRACT 2: vision_token_rep
              │                   (at IMG1 position, layers 0,8,16,24,31)
              │                   5 × [3072]
              │                             │
              └──────────┬──────────────────┘
                         │
                         ▼
                   ┌──────────────┐
                   │  Generation  │
                   └──────┬───────┘
                          │
                          ▼
                   EXTRACT 4: answer (text)
```

---

## Code Location Reference

| Function | Lines | Purpose |
|----------|-------|---------|
| `extract_embeddings` | 122-148 | Main orchestration |
| `_generate_answer` | 150-164 | Generate answer text |
| `_extract_vision_representation` | 166-195 | Vision-only features |
| `_extract_decoder_embeddings` | 197-236 | Vision & query token features |
| `_target_layers` | 111-120 | Select which layers to extract |

This architecture allows probing the model at multiple stages to understand:
1. What it sees (vision_only)
2. How it integrates vision and language (vision_token at different depths)
3. How it understands the question+image (query_token at different depths)
4. What it predicts (answer)

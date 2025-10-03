# Phi-4 Concrete Extraction Examples

This document shows exactly what data structures are extracted and their specific shapes/values.

## Example VQA Input

**Image:** `haloquest_1236.png`
**Question:** `"Is the forest lit by a full moonlight making the mushrooms glow brighter than usual?"`
**Ground Truth:** `"No; The forest isn't brightened by full moonlight; There is no glowing mushroom in the forest"`

---

## Step-by-Step Extraction with Actual Data

### Step 1: Input Processing

```python
# Line 126: Create Phi-4 formatted prompt
prompt = f'<|user|><|image_1|>{question}<|end|><|assistant|>'

# Result:
prompt = "<|user|><|image_1|>Is the forest lit by a full moonlight making the mushrooms glow brighter than usual?<|end|><|assistant|>"
```

```python
# Line 128-132: Process into model inputs
inputs = processor(
    text=prompt,
    images=[image],
    return_tensors="pt"
).to(device)

# Result:
inputs = {
    'input_ids': tensor([[29871, 29989, 1792, 29989, 200010, ...]]),
    #                                            ^^^^^^
    #                                            <|image_1|> token at position varies

    'attention_mask': tensor([[1, 1, 1, ..., 1]]),  # Length varies

    'pixel_values': tensor([[[...]]]),  # Image tensor
    # ... other image-related tensors from processor
}
```

---

## Extraction Point 1: Vision-Only Representation

### Code Flow

```python
# Line 138: Call extraction
vision_only_rep = self._extract_vision_representation(image)

# Line 174: Use image processor
vis_inputs = self.processor.image_processor(images=[image], return_tensors="pt")

# Phi-4's processor returns pre-computed vision embeddings
# Line 177-180: Check for pre-computed embeddings
if 'input_image_embeds' in vis_inputs:
    feats = vis_inputs['input_image_embeds'].to(self.model.device)
    # feats shape varies based on image size and tiling
```

### Pooling to Single Vector

```python
# Line 183-184: Pool across all dimensions except the last (feature dim)
while feats.dim() > 2:
    feats = feats.mean(dim=1)

# Start: [batch, tiles/patches, features]
# After pooling: [batch, features]

# Line 187: Final result
rep = feats.squeeze(0).to(torch.float32).cpu().numpy()  # Shape: (448,)
```

### Output

```python
vision_only_representation = array([
    -0.08390875, -0.08510972, -0.0862821,  -0.08775678, -0.09075864,
    -0.09185518, -0.0932598,  -0.09061275, -0.08737629, -0.08777604,
    ...
    -0.06084035, -0.05558707, -0.05234944, -0.05261905, -0.05837302,
    -0.07107785, -0.08228059, -0.08293185, -0.08795401, -0.09263831
])
# Shape: (448,)
# Mean: -0.029682
# Std: 0.029898
```

**What this represents:**
- A single 448-dimensional vector from SigLIP vision encoder
- Pure visual features before any language processing
- Averaged across all spatial positions and tiles

---

## Extraction Points 2 & 3: Vision Token and Query Token Representations

### Code Flow

```python
# Line 141: Call extraction for both
vision_token_reps, query_token_reps = self._extract_decoder_embeddings(
    inputs,
    self._target_layers()  # Returns [0, 8, 16, 24, 31]
)

# Line 199-200: Forward pass through FULL model
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)

# Result:
outputs.hidden_states = tuple of 33 tensors:
    [0]: embeddings         [1, seq_len, 3072]  (token embeddings)
    [1]: layer_0_output     [1, seq_len, 3072]  (after 1st transformer layer)
    [2]: layer_1_output     [1, seq_len, 3072]
    ...
    [9]: layer_8_output     [1, seq_len, 3072]  (25% through)
    ...
    [17]: layer_16_output   [1, seq_len, 3072]  (50% through)
    ...
    [25]: layer_24_output   [1, seq_len, 3072]  (75% through)
    ...
    [32]: layer_31_output   [1, seq_len, 3072]  (final layer)
```

### Finding Token Boundaries

```python
# Line 206-208: Get input_ids on CPU
input_ids = inputs["input_ids"][0]  # Shape: [seq_len]

# Line 211-213: Find image token position
image_positions = (input_ids == 200010).nonzero()  # 200010 = <|image_1|>
vision_token_boundary = int(image_positions[-1].item())
# Result: vision_token_boundary = 2625 (example from actual run)

# Line 218-222: Find last token position
last_idx = int(attention_mask[0].sum().item()) - 1
query_token_boundary = last_idx
# Result: query_token_boundary = 2640 (example from actual run)
```

### Actual Sequence with Positions

```
Position:  0      1      2     3      ...  2625      ...  2640
Token:    [user] [delim] ...  ...     ... [IMG1]    ...  [asst]
ID:       29871  29989   ...  ...     ... 200010    ...  465
                                          ^                ^
                              vision_token_boundary   query_token_boundary
```

**Note:** Phi-4 sequences can be quite long because:
1. Image embeddings are injected at the `<|image_1|>` position
2. The vision features expand into multiple tokens internally
3. This is why vision_token_boundary = 2625 (not just position 4)

### Extracting Hidden States

```python
# Line 228-234: Extract for each layer
vision_token_reps = {}
query_token_reps = {}

for k in [0, 8, 16, 24, 31]:  # target_layers
    # Get hidden state from layer k
    # Note: hidden_states[k+1] because hidden_states[0] is embeddings
    layer_h = hidden_states[k + 1][0]  # [seq_len, 3072]

    # Extract at vision token position
    vt_vec = layer_h[vision_token_boundary]  # [3072]
    vision_token_reps[f"layer_{k}"] = vt_vec.cpu().numpy()

    # Extract at query token position
    qt_vec = layer_h[query_token_boundary]  # [3072]
    query_token_reps[f"layer_{k}"] = qt_vec.cpu().numpy()
```

### Output

```python
vision_token_representation = {
    'layer_0': array([
        0.02490234,  0.14648438, -0.23242188,  0.12255859, -0.01391602,
        ...  # 3072 values total
    ]),  # Shape: (3072,)

    'layer_8': array([
        0.26953125,  0.765625,    0.23632812, -0.05541992,  0.01257324,
        ...
    ]),  # Shape: (3072,)

    'layer_16': array([
        0.26953125,  1.1015625,   1.640625,   -0.8125,     -0.29296875,
        ...
    ]),  # Shape: (3072,)

    'layer_24': array([
        0.41015625,  1.40625,     3.125,       0.52734375, -0.18261719,
        ...
    ]),  # Shape: (3072,)

    'layer_31': array([
        0.34375,     0.7109375,   0.81640625,  0.29882812, -0.43359375,
        ...
    ]),  # Shape: (3072,)
}

query_token_representation = {
    'layer_0': array([
        -0.02331543,  0.00537109,  0.03857422, -0.14160156, -0.02734375,
        ...
    ]),  # Shape: (3072,)

    'layer_8': array([
        -0.06347656,  0.3046875,   0.09570312, -0.22167969, -0.05712891,
        ...
    ]),  # Shape: (3072,)

    'layer_16': array([
        0.17578125,  0.73046875,  0.265625,   -0.4296875,   0.10058594,
        ...
    ]),  # Shape: (3072,)

    'layer_24': array([
        -0.21875,     0.61328125, -3.0,         1.578125,   -1.2890625,
        ...
    ]),  # Shape: (3072,)

    'layer_31': array([
        0.03198242,  0.58203125,  0.48632812,  0.48828125,  0.49804688,
        ...
    ]),  # Shape: (3072,)
}
```

**Statistics for each layer:**

| Layer | Vision Token | Query Token |
|-------|--------------|-------------|
| **layer_0** | mean=-0.0025, std=0.348 | mean=-0.0016, std=0.152 |
| **layer_8** | mean=-0.0024, std=0.672 | mean=-0.0137, std=0.423 |
| **layer_16** | mean=-0.0266, std=1.076 | mean=-0.0458, std=1.055 |
| **layer_24** | mean=-0.0474, std=1.989 | mean=-0.0771, std=2.443 |
| **layer_31** | mean=0.0080, std=1.042 | mean=-0.0168, std=1.040 |

**Observations:**
- Standard deviation increases with depth (layer 0: ~0.35 → layer 24: ~2.0)
- This shows increasing specialization and task-specific features
- Query tokens have different statistics than vision tokens (specialization)

**What these represent:**
- **vision_token_reps**: How the model represents the image at different depths
  - At position 2625 (where `<|image_1|>` token is embedded)
  - Shows image understanding as it's processed through the decoder

- **query_token_reps**: How the model represents the question+image together
  - At position 2640 (end of prompt, before generation)
  - Shows integrated multimodal understanding at different depths

---

## Extraction Point 4: Generated Answer

### Code Flow

```python
# Line 135: Generate answer
generated_text = self._generate_answer(inputs)

# Line 152-157: Generate tokens with mixed precision
with torch.autocast(device_type="cuda", dtype=torch.float16):
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False  # Greedy decoding (argmax)
    )

# Result:
outputs.shape = torch.Size([1, seq_len + N])  # Input tokens + generated tokens
# outputs[0] = [29871, 29989, ..., 200010, ..., generated_token_1, generated_token_2, ...]
#               └──────────── input ──────────┘ └──────── generated ────────┘

# Line 159: Skip the input tokens
gen_tokens = outputs[0, inputs["input_ids"].size(1):]  # Only the newly generated part

# Line 160: Decode to text
text = processor.decode(gen_tokens, skip_special_tokens=True)
```

### Output

```python
answer = "The image does not provide any information about the source of light,
          such as a full moon, so it cannot be determined if the mushrooms are
          glowing due to moonlight."
```

**What this represents:**
- The model's final answer to the VQA question
- Result of full multimodal reasoning and generation

---

## Complete Output Structure

When saved to HDF5, each question gets this structure:

```python
question_comb_1732/
├── image_id: "haloquest_1236.png"
├── question: "Is the forest lit by a full moonlight making the mushrooms glow brighter than usual?"
├── ground_truth_answer: "No; The forest isn't brightened by full moonlight; There is no glowing mushroom in the forest"
│
├── vision_only_representation: [448]
│   └── Single pooled SigLIP vision vector
│
├── vision_token_representation/
│   ├── layer_0:  [3072]  ← Hidden state at position 2625, layer 0
│   ├── layer_8:  [3072]  ← Hidden state at position 2625, layer 8
│   ├── layer_16: [3072]  ← Hidden state at position 2625, layer 16
│   ├── layer_24: [3072]  ← Hidden state at position 2625, layer 24
│   └── layer_31: [3072]  ← Hidden state at position 2625, layer 31
│
├── query_token_representation/
│   ├── layer_0:  [3072]  ← Hidden state at position 2640, layer 0
│   ├── layer_8:  [3072]  ← Hidden state at position 2640, layer 8
│   ├── layer_16: [3072]  ← Hidden state at position 2640, layer 16
│   ├── layer_24: [3072]  ← Hidden state at position 2640, layer 24
│   └── layer_31: [3072]  ← Hidden state at position 2640, layer 31
│
└── answer: "The image does not provide any information about the source..."
```

**Total data per sample:**
- 1 × 448-dim vector (vision_only)
- 5 × 3072-dim vectors (vision_token, 5 layers)
- 5 × 3072-dim vectors (query_token, 5 layers)
- 1 text string (answer)

---

## Position-Specific Extraction Visualization

```
Phi-4 Input Sequence (2641 tokens after embedding expansion):

Index:     0      1      2      3     ...  2625     ...  2640
Token:   [user] [delim] [...]  [...] ... [IMG1]   ...  [asst]
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Hidden States from Language Decoder:

Layer 0:   h₀₀   h₀₁   h₀₂   h₀₃  ... h₀,₂₆₂₅  ... h₀,₂₆₄₀
                                           ↑            ↑
                                           │            │
Layer 8:   h₈₀   h₈₁   h₈₂   h₈₃  ... h₈,₂₆₂₅  ... h₈,₂₆₄₀
                                           ↑            ↑
                                           │            │
Layer 16:  h₁₆₀  h₁₆₁  h₁₆₂  h₁₆₃ ... h₁₆,₂₆₂₅ ... h₁₆,₂₆₄₀
                                           ↑            ↑
                                           │            │
Layer 24:  h₂₄₀  h₂₄₁  h₂₄₂  h₂₄₃ ... h₂₄,₂₆₂₅ ... h₂₄,₂₆₄₀
                                           ↑            ↑
                                           │            │
Layer 31:  h₃₁₀  h₃₁₁  h₃₁₂  h₃₁₃ ... h₃₁,₂₆₂₅ ... h₃₁,₂₆₄₀
                                           ↑            ↑
                                           │            │
                         vision_token_boundary   query_token_boundary
                              (position 2625)      (position 2640)
                                           │            │
                                           ▼            ▼
                         EXTRACT: h₀,₂₆₂₅, h₈,₂₆₂₅,   EXTRACT: h₀,₂₆₄₀, h₈,₂₆₄₀,
                                  h₁₆,₂₆₂₅, h₂₄,₂₆₂₅,          h₁₆,₂₆₄₀, h₂₄,₂₆₄₀,
                                  h₃₁,₂₆₂₅                     h₃₁,₂₆₄₀
                                           │            │
                                           ▼            ▼
                         vision_token_representation  query_token_representation
                               (5 × [3072])                (5 × [3072])
```

Each extracted vector is a 3072-dimensional representation at that specific position and layer.

---

## Why These Specific Positions?

### Vision Token (Position 2625 - `<|image_1|>`)
- This position is where the vision embeddings are **injected** into the language sequence
- The actual `<|image_1|>` token (ID 200010) gets replaced/augmented with projected vision features
- The position is far into the sequence because:
  1. Phi-4's template includes `<|user|>` prefix
  2. Vision embeddings expand into multiple positions internally
  3. The processor handles this injection automatically

### Query Token (Position 2640 - End of Prompt)
- This position is at `<|assistant|>` token - right before generation starts
- It contains the full context: user prompt + image + question + end markers
- This is the **integrated multimodal understanding** right before the model starts answering
- The hidden state here is what gets fed to the LM head to predict the first answer token

---

## Layer Progression Interpretation

As information flows through the 32 layers:

**Layer 0 (0%):**
- Initial embedding with vision injection
- Raw multimodal fusion
- Std: ~0.35 (relatively constrained)

**Layer 8 (25%):**
- Early semantic processing
- Starting abstraction
- Std: ~0.67 (increasing variance)

**Layer 16 (50%):**
- Mid-level reasoning
- Multimodal concepts well-formed
- Std: ~1.08 (more specialized)

**Layer 24 (75%):**
- High-level semantic understanding
- Task-specific features emerging
- Std: ~1.99 (highly specialized)

**Layer 31 (100%):**
- Final representation
- Ready for answer generation
- Std: ~1.04 (stabilized for generation)

By extracting from these layers, you can probe how the model's understanding evolves during processing.

---

## Comparison with Llama 3.2 Output

| Aspect | Llama 3.2 | Phi-4 |
|--------|-----------|-------|
| Vision-only dim | 7680 | 448 |
| Language hidden | 4096 | 3072 |
| Layers sampled | [0,10,20,30,39] | [0,8,16,24,31] |
| Total layers | 40 | 32 |
| Vision token position | ~6 | ~2625 |
| Query token position | ~33 | ~2640 |
| Sequence length | Short (~34) | Long (~2641) |

**Key difference:** Phi-4 has MUCH longer sequences because vision embeddings are expanded internally into many tokens, while Llama 3.2 uses a more compact representation.

---

## Real Statistics from Test Run

From the actual HDF5 output:

### Sample 1: question_comb_1732
- Image: `haloquest_1236.png`
- Vision-only: shape=(448,), mean=-0.0297, std=0.0299
- Vision token (layer_31): mean=0.0080, std=1.042
- Query token (layer_31): mean=-0.0168, std=1.040
- Cosine similarity (layer_31): 0.673

### Sample 2: question_comb_4685
- Image: `mme_celebrity_tt0064115_shot_0367_img_0.jpg`
- Vision-only: shape=(448,), mean=-0.2775, std=0.0684
- Vision token (layer_31): mean=-0.0015, std=1.026
- Query token (layer_31): mean=-0.0039, std=0.486
- Cosine similarity (layer_31): 0.616

### Sample 3: question_comb_6253
- Image: `haloquest_2016.png`
- Vision-only: shape=(448,), mean=0.0154, std=0.1120
- Vision token (layer_31): mean=0.0080, std=1.048
- Query token (layer_31): mean=-0.0159, std=0.941
- Cosine similarity (layer_31): 0.584

**Pattern observed:** Vision and query tokens maintain reasonable cosine similarity (~0.6-0.7 at final layer), but are distinct enough to capture different aspects of the multimodal understanding.

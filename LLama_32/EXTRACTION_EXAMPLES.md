# Concrete Extraction Examples

This document shows exactly what data structures are extracted and their specific shapes/values.

## Example VQA Input

**Image:** `haloquest_1236.png`
**Question:** `"Is the forest lit by a full moon?"`
**Ground Truth:** `"No; The forest isn't brightened by full moonlight"`

---

## Step-by-Step Extraction with Actual Data

### Step 1: Input Processing

```python
# Line 115-124: Create chat-formatted prompt
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Is the forest lit by a full moon?"}
        ]
    }
]
chat_text = processor.apply_chat_template(messages, add_generation_prompt=True)

# Result:
chat_text = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

<|image|>Is the forest lit by a full moon?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
```

```python
# Line 126-130: Process into model inputs
inputs = processor(
    text=chat_text,
    images=[image],
    return_tensors="pt"
).to(device)

# Result:
inputs = {
    'input_ids': tensor([[128000, 128006, 9125, 128007, ..., 128256, ..., 30, 128009, 128007]]),
    #                                                        ^^^^^^
    #                                                        <|image|> token at position 6

    'attention_mask': tensor([[1, 1, 1, ..., 1, 1, 1]]),  # Length 34

    'pixel_values': tensor([[[[[...]]]]]),  # Shape: [1, 1, 4, 3, 560, 560]
    'aspect_ratio_ids': tensor([[1]]),
    'aspect_ratio_mask': tensor([[[1, 0, 0, 0]]])
}
```

---

## Extraction Point 1: Vision-Only Representation

### Code Flow

```python
# Line 136: Call extraction
vision_only_rep = self._extract_vision_representation(image)

# Line 170-171: Prepare vision-only inputs
vis_inputs = processor(images=[image], return_tensors="pt")
vis_inputs = {k: v.to(device) for k, v in vis_inputs.items()}
# vis_inputs contains: pixel_values, aspect_ratio_ids, aspect_ratio_mask

# Line 174: Forward pass through ONLY the vision encoder
vis_out = self._vision_submodule(**vis_inputs)
# self._vision_submodule = model.vision_model (MllamaVisionModel)

# Result:
vis_out.last_hidden_state.shape = torch.Size([1, 1, 4, 1601, 7680])
#                                             │  │  │  │     └─ hidden dimension
#                                             │  │  │  └─ sequence length (patches)
#                                             │  │  └─ number of tiles (4 for this image)
#                                             │  └─ number of images
#                                             └─ batch size
```

### Shape Breakdown

The vision encoder processes the image as **4 tiles** (because of the aspect ratio):

```
Original Image (e.g., 800×600)
         ↓
Tiled into 4 regions of 560×560
         ↓
Each tile → ViT → sequence of 1601 patches
         ↓
Each patch → 7680-dimensional vector

Total: [1 batch, 1 image, 4 tiles, 1601 patches, 7680 dims]
```

### Pooling to Single Vector

```python
# Line 182-186: Pool across all dimensions
feats = vis_out.last_hidden_state  # [1, 1, 4, 1601, 7680]

while feats.dim() > 1:
    feats = feats.mean(dim=0)

# Iteration 1: mean over batch → [1, 4, 1601, 7680]
# Iteration 2: mean over num_images → [4, 1601, 7680]
# Iteration 3: mean over tiles → [1601, 7680]
# Iteration 4: mean over sequence → [7680]

# Line 189: Final result
rep = feats.to(torch.float32).cpu().numpy()  # Shape: (7680,)
```

### Output

```python
vision_only_representation = array([
    -0.00631,  0.01203, -0.00845, ..., -0.01234, 0.00567, -0.00892
])
# Shape: (7680,)
# Mean: -0.0063
# Std: 1.5741
```

**What this represents:**
- A single 7680-dimensional vector summarizing the entire image
- Pure visual features before any language processing
- Averaged across all tiles and spatial positions

---

## Extraction Points 2 & 3: Vision Token and Query Token Representations

### Code Flow

```python
# Line 139: Call extraction for both
vision_token_reps, query_token_reps = self._extract_decoder_embeddings(
    inputs,
    self._target_layers()  # Returns [0, 10, 20, 30, 39]
)

# Line 198: Forward pass through FULL model
outputs = model(**inputs, output_hidden_states=True, return_dict=True)

# Result:
outputs.hidden_states = tuple of 41 tensors:
    [0]: embeddings         [1, 34, 4096]  (token embeddings)
    [1]: layer_0_output     [1, 34, 4096]  (after 1st transformer layer)
    [2]: layer_1_output     [1, 34, 4096]
    ...
    [11]: layer_10_output   [1, 34, 4096]  (25% through)
    ...
    [21]: layer_20_output   [1, 34, 4096]  (50% through)
    ...
    [31]: layer_30_output   [1, 34, 4096]  (75% through)
    ...
    [40]: layer_39_output   [1, 34, 4096]  (final layer)
```

### Finding Token Boundaries

```python
# Line 204-206: Get input_ids on CPU
input_ids = inputs["input_ids"][0]  # Shape: [34]
# input_ids = [128000, 128006, 9125, 128007, 271, 128256, 3957, 279, 13952, ...]
#              [  0  ] [  1  ] [ 2 ] [  3  ] [ 4] [  5  ] [ 6 ] [ 7 ] [  8 ] ...

# Line 209-211: Find image token position
image_positions = (input_ids == 128256).nonzero()  # 128256 = <|image|>
# Result: tensor([5])  (position where <|image|> token is)
vision_token_boundary = 5

# Line 216-220: Find last token position
last_idx = int(attention_mask[0].sum().item()) - 1
# attention_mask.sum() = 34 (number of non-padded tokens)
# last_idx = 33
query_token_boundary = 33
```

### Actual Sequence with Positions

```
Position:  0      1      2      3     4   5       6    7   8      9   ...  33
Token:    [BOS] [USER] [HEAD] [EOT] [\n][IMG] [Is] [the][forest][lit] ... [ASST]
ID:       128000 128006 9125  128007 271 128256 3957 279 13952  13   ... 128007
                                        ^                                   ^
                              vision_token_boundary = 5        query_token_boundary = 33
```

### Extracting Hidden States

```python
# Line 226-232: Extract for each layer
vision_token_reps = {}
query_token_reps = {}

for k in [0, 10, 20, 30, 39]:  # target_layers
    # Get hidden state from layer k
    # Note: hidden_states[k+1] because hidden_states[0] is embeddings
    layer_h = hidden_states[k + 1][0]  # [34, 4096]

    # Extract at vision token position (index 5)
    vt_vec = layer_h[5]  # [4096]
    vision_token_reps[f"layer_{k}"] = vt_vec.cpu().numpy()

    # Extract at query token position (index 33)
    qt_vec = layer_h[33]  # [4096]
    query_token_reps[f"layer_{k}"] = qt_vec.cpu().numpy()
```

### Output

```python
vision_token_representation = {
    'layer_0':  array([0.123, -0.456, 0.789, ..., -0.234]),  # Shape: (4096,)
    'layer_10': array([0.234, -0.567, 0.890, ..., -0.345]),  # Shape: (4096,)
    'layer_20': array([0.345, -0.678, 0.901, ..., -0.456]),  # Shape: (4096,)
    'layer_30': array([0.456, -0.789, 0.012, ..., -0.567]),  # Shape: (4096,)
    'layer_39': array([0.567, -0.890, 0.123, ..., -0.678]),  # Shape: (4096,)
}

query_token_representation = {
    'layer_0':  array([0.321, -0.654, 0.987, ..., -0.432]),  # Shape: (4096,)
    'layer_10': array([0.432, -0.765, 0.098, ..., -0.543]),  # Shape: (4096,)
    'layer_20': array([0.543, -0.876, 0.109, ..., -0.654]),  # Shape: (4096,)
    'layer_30': array([0.654, -0.987, 0.210, ..., -0.765]),  # Shape: (4096,)
    'layer_39': array([0.765, -0.098, 0.321, ..., -0.876]),  # Shape: (4096,)
}
```

**What these represent:**
- **vision_token_reps**: How the model represents the image at different depths of processing
  - At position 5 (where `<|image|>` token is)
  - Shows image understanding as it's processed through the decoder

- **query_token_reps**: How the model represents the question+image together
  - At position 33 (end of prompt, before generation)
  - Shows integrated multimodal understanding at different depths

---

## Extraction Point 4: Generated Answer

### Code Flow

```python
# Line 133: Generate answer
generated_text = self._generate_answer(inputs)

# Line 151-155: Generate tokens
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=False  # Greedy decoding (argmax)
)

# Result:
outputs.shape = torch.Size([1, 34 + N])  # Input tokens + generated tokens
# outputs[0] = [128000, 128006, ..., 128256, ..., 128007, 791, 2217, 1587, ...]
#               └──────────── input (34) ──────────┘ └──── generated (N) ────┘

# Line 157: Skip the input tokens
gen_tokens = outputs[0, 34:]  # Only the newly generated part

# Line 158: Decode to text
text = processor.decode(gen_tokens, skip_special_tokens=True)
```

### Output

```python
answer = "The image does not appear to be lit by a full moon. The lighting in the
          image is more blue and green, and the mushrooms are glowing with a soft,
          ethereal light. The overall atmosphere is dreamy and surreal, with the
          mushrooms and the girl standing in a misty, mystical forest."
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
├── question: "Is the forest lit by a full moon?"
├── ground_truth_answer: "No; The forest isn't brightened by full moonlight"
│
├── vision_only_representation: [7680]
│   └── Single pooled vision vector
│
├── vision_token_representation/
│   ├── layer_0:  [4096]  ← Hidden state at position 5, layer 0
│   ├── layer_10: [4096]  ← Hidden state at position 5, layer 10
│   ├── layer_20: [4096]  ← Hidden state at position 5, layer 20
│   ├── layer_30: [4096]  ← Hidden state at position 5, layer 30
│   └── layer_39: [4096]  ← Hidden state at position 5, layer 39
│
├── query_token_representation/
│   ├── layer_0:  [4096]  ← Hidden state at position 33, layer 0
│   ├── layer_10: [4096]  ← Hidden state at position 33, layer 10
│   ├── layer_20: [4096]  ← Hidden state at position 33, layer 20
│   ├── layer_30: [4096]  ← Hidden state at position 33, layer 30
│   └── layer_39: [4096]  ← Hidden state at position 33, layer 39
│
└── answer: "The image does not appear to be lit by a full moon..."
```

**Total data per sample:**
- 1 × 7680-dim vector (vision_only)
- 5 × 4096-dim vectors (vision_token, 5 layers)
- 5 × 4096-dim vectors (query_token, 5 layers)
- 1 text string (answer)

---

## Position-Specific Extraction Visualization

```
Input Sequence (34 tokens):

Index:     0    1    2    3    4    5      6    7    8    9   ...  33
Token:   [BOS][USR][HDR][EOT][\n][IMG] [Is][the][for][lit] ... [ASST]
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Hidden States from Language Decoder:

Layer 0:   h₀₀  h₀₁  h₀₂  h₀₃  h₀₄  h₀₅   h₀₆  h₀₇  h₀₈  h₀₉ ... h₀₃₃
                                    ↑                              ↑
                                    │                              │
Layer 10:  h₁₀  h₁₁  h₁₂  h₁₃  h₁₄  h₁₅   h₁₆  h₁₇  h₁₈  h₁₉ ... h₁₃₃
                                    ↑                              ↑
                                    │                              │
Layer 20:  h₂₀  h₂₁  h₂₂  h₂₃  h₂₄  h₂₅   h₂₆  h₂₇  h₂₈  h₂₉ ... h₂₃₃
                                    ↑                              ↑
                                    │                              │
Layer 30:  h₃₀  h₃₁  h₃₂  h₃₃  h₃₄  h₃₅   h₃₆  h₃₇  h₃₈  h₃₉ ... h₃₃₃
                                    ↑                              ↑
                                    │                              │
Layer 39:  h₃₉₀ h₃₉₁ h₃₉₂ h₃₉₃ h₃₉₄ h₃₉₅  h₃₉₆ h₃₉₇ h₃₉₈ h₃₉₉ ... h₃₉₃₃
                                    ↑                              ↑
                                    │                              │
                         vision_token_boundary          query_token_boundary
                              (position 5)                  (position 33)
                                    │                              │
                                    ▼                              ▼
                         EXTRACT: h₀₅, h₁₅, h₂₅,        EXTRACT: h₀₃₃, h₁₃₃, h₂₃₃,
                                  h₃₅, h₃₉₅                      h₃₃₃, h₃₉₃₃
                                    │                              │
                                    ▼                              ▼
                         vision_token_representation    query_token_representation
                               (5 × [4096])                   (5 × [4096])
```

Each extracted vector is a 4096-dimensional representation at that specific position and layer.

---

## Why These Specific Positions?

### Vision Token (Position 5 - `<|image|>`)
- This position is where the cross-attention mechanism **injects** the processed image information into the language sequence
- The hidden state here captures how the model internally represents the image content
- As it progresses through layers, this representation gets refined with linguistic context

### Query Token (Position 33 - End of Prompt)
- This position is the "query" that will be used to start generation
- It contains the full context: system prompt + image + question
- This is the **integrated multimodal understanding** right before the model starts answering
- The hidden state here is what gets fed to the LM head to predict the first answer token

---

## Layer Progression Interpretation

As information flows through the 40 layers:

**Layer 0 (0%):**
- Initial fusion of vision and text
- Still relatively "raw" multimodal features

**Layer 10 (25%):**
- Early semantic processing
- Starting to form abstractions

**Layer 20 (50%):**
- Mid-level reasoning
- Multimodal concepts well-formed

**Layer 30 (75%):**
- High-level semantic understanding
- Preparing for generation

**Layer 39 (100%):**
- Final representation
- Ready for answer generation
- Most "task-specific" features

By extracting from these layers, you can probe how the model's understanding evolves during processing.

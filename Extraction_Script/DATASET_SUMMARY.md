# Dataset Summary - Composite Multi-Source VQA for Hallucination Detection

## Quick Reference

**Dataset:** `sampled_10k_relational_dataset.csv`  
**Total Samples:** 10,000  
**Relational Questions:** 70%  
**Source Datasets:** 6 (AMBER, HaloQuest, POPE, MME, HallusionBench, MathVista)

---

## Source Breakdown

| Dataset | Samples | % | Focus |
|---------|---------|---|-------|
| **AMBER** | 3,926 | 39.26% | Discriminative tasks, attributes, relations |
| **HaloQuest** | 2,784 | 27.84% | Adversarial hallucination challenges |
| **POPE** | 1,230 | 12.30% | Object hallucination (random/adversarial) |
| **MME** | 885 | 8.85% | Multi-modal reasoning, knowledge |
| **HallusionBench** | 617 | 6.17% | Visual illusions, video understanding |
| **MathVista** | 558 | 5.58% | Mathematical visual reasoning |

---

## Top 15 Question Categories

| Rank | Category | Samples | % | Source |
|------|----------|---------|---|--------|
| 1 | Visual Challenge | 1,531 | 15.31% | HaloQuest |
| 2 | Discriminative Attribute State | 1,169 | 11.69% | AMBER |
| 3 | Discriminative Relation | 975 | 9.75% | AMBER |
| 4 | False Premises | 898 | 8.98% | HaloQuest |
| 5 | Relation | 689 | 6.89% | AMBER |
| 6 | Discriminative Hallucination | 620 | 6.20% | AMBER |
| 7 | Random | 456 | 4.56% | POPE |
| 8 | Adversarial | 402 | 4.02% | POPE |
| 9 | Popular | 372 | 3.72% | POPE |
| 10 | Insufficient Context | 355 | 3.55% | HaloQuest |
| 11 | Celebrity | 340 | 3.40% | MME |
| 12 | Math-Targeted VQA | 323 | 3.23% | MathVista |
| 13 | Attribute Number | 280 | 2.80% | AMBER |
| 14 | General VQA | 235 | 2.35% | MathVista |
| 15 | Video | 170 | 1.70% | HallusionBench |

**Total Categories:** 38 unique types

---

## Key Characteristics

### Diversity
- ✅ 6 established VQA benchmarks
- ✅ 38 question categories
- ✅ Real + generated images
- ✅ Multiple difficulty levels

### Hallucination Focus
- ✅ 28% from hallucination-specific dataset (HaloQuest)
- ✅ 12% object hallucination tests (POPE)
- ✅ Adversarial and false premise questions
- ✅ Ambiguous/insufficient context scenarios

### Relational Emphasis
- ✅ 70% relational reasoning
- ✅ Spatial relations (contact, position)
- ✅ Attribute relations (which X has Y?)
- ✅ Temporal relations (video sequences)

### Quality
- ✅ 100% image availability
- ✅ Manual hallucination annotations
- ✅ Multiple ground truth phrasings
- ✅ Standardized CSV format

---

## Dataset Statistics by Model

All 8 VLMs evaluated on same 10k dataset with **manual hallucination annotations**:

| Model | Manual Review CSV |
|-------|-------------------|
| Gemma3-12B | gemma3_manually_reviewed.csv |
| FastVLM-7B | fastvlm_manually_reviewed.csv |
| LLaVA-Next-8B | llava_manually_reviewed.csv |
| Molmo-V1 | molmo_manually_reviewed.csv |
| Qwen2.5-VL-7B | qwen25vl_manually_reviewed.csv |
| Llama-3.2-11B | llama32_manually_reviewed.csv |
| Phi4-VL | phi4vl_manually_reviewed.csv |
| SmolVLM2-2.2B | smolvlm_manually_reviewed.csv |

**Hallucination Distribution:** ~90% no hallucination, ~10% hallucination

---

## Example Questions

### Visual Challenge (15.31%)
```
Q: How many fingers does the human have including the thumb?
A: Six fingers
Category: Anatomical error detection (generated images)
```

### False Premises (8.98%)
```
Q: How many sharks are present in the travel brochure?
A: There are no sharks in the brochure
Category: Rejecting incorrect assumptions
```

### Discriminative Relation (9.75%)
```
Q: Is there direct contact between the dog and grass?
A: Yes
Category: Spatial relationship reasoning
```

### Adversarial (4.02%)
```
Q: Is there a dining table in the image?
A: No (challenging context)
Category: Object hallucination test
```

### Math-Targeted VQA (3.23%)
```
Q: The line is about (_) centimeters long.
A: 7
Category: Quantitative visual reasoning
```

---

## Dataset Files

```
/root/akhil/final_data/
├── sampled_10k_relational_dataset.csv       # Primary dataset
└── all_images/                              # 10k images

/root/akhil/FInal_CSV_Hallucination/
├── gemma3_manually_reviewed.csv
├── fastvlm_manually_reviewed.csv
├── llava_manually_reviewed.csv
├── molmo_manually_reviewed.csv
├── qwen25vl_manually_reviewed.csv
├── llama32_manually_reviewed.csv
├── phi4vl_manually_reviewed.csv
└── smolvlm_manually_reviewed.csv
```

---

## Why This Composition?

1. **Scientific Rigor:** Established benchmarks, published protocols
2. **Hallucination Focus:** 40%+ adversarial/challenging questions
3. **Real-World Relevance:** Practical scenarios (celebrities, math, scenes)
4. **Comprehensive Coverage:** 38 categories prevent overfitting
5. **Relational Emphasis:** 70% relational reasoning (understudied)
6. **Reproducibility:** Standardized format, public source datasets

---

## Citation

When using this dataset, cite the source benchmarks:

- **AMBER:** Wang et al., "AMBER: An LLM-free Multi-dimensional Benchmark for MLLMs Hallucination Evaluation"
- **HaloQuest:** Custom hallucination-focused dataset
- **POPE:** Li et al., "Evaluating Object Hallucination in Large Vision-Language Models"
- **MME:** Fu et al., "MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models"
- **HallusionBench:** Liu et al., "HallusionBench: You See What You Think?"
- **MathVista:** Lu et al., "MathVista: Evaluating Math Reasoning in Visual Contexts"

---

**Last Updated:** October 4, 2025  
**Version:** 1.0  
**Total Samples:** 10,000  
**Models Evaluated:** 8 VLMs

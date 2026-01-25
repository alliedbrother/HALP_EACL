# Question and Answer Type Examples for VQA Dataset

This document provides detailed examples for each question and answer type category to improve diversity in the VQA hallucination detection dataset.

---

## 1. Numeric/Quantitative Answers (Target: 15-20%)

### 1.1 Exact Count

**Example 1:**
- **Question:** How many dogs are visible in the image?
- **Answer:** 3

**Example 2:**
- **Question:** How many windows are on the front of the building?
- **Answer:** 8

---

### 1.2 Approximate Count

**Example 1:**
- **Question:** About how many people are in the crowd?
- **Answer:** approximately 50

**Example 2:**
- **Question:** Roughly how many trees are visible in the forest?
- **Answer:** around 20-25

---

### 1.3 Measurements

**Example 1:**
- **Question:** How tall is the building approximately?
- **Answer:** approximately 10 stories

**Example 2:**
- **Question:** What is the approximate width of the table?
- **Answer:** about 6 feet

---

### 1.4 Percentages

**Example 1:**
- **Question:** What percentage of the image is occupied by sky?
- **Answer:** about 30%

**Example 2:**
- **Question:** Approximately what percentage of the plate is covered with food?
- **Answer:** roughly 75%

---

### 1.5 Comparative Counts

**Example 1:**
- **Question:** How many more cats are there than dogs?
- **Answer:** 2 more cats

**Example 2:**
- **Question:** Are there more people sitting or standing?
- **Answer:** 3 more people standing

---

## 2. Multiple Choice/Selection (Target: 10-15%)

### 2.1 Option Selection

**Example 1:**
- **Question:** Which color is the car? A) Red B) Blue C) Green D) Yellow
- **Answer:** B

**Example 2:**
- **Question:** What is the weather in the image? A) Sunny B) Rainy C) Snowy D) Cloudy
- **Answer:** A

---

### 2.2 Best Description

**Example 1:**
- **Question:** Which best describes the scene? A) Indoor B) Outdoor C) Urban D) Rural
- **Answer:** C

**Example 2:**
- **Question:** What type of activity is shown? A) Sports B) Dining C) Working D) Relaxing
- **Answer:** B

---

### 2.3 Ranked Selection

**Example 1:**
- **Question:** What is the primary object in the foreground? A) Tree B) Car C) Person
- **Answer:** C

**Example 2:**
- **Question:** Which dominates the color palette? A) Blue tones B) Warm tones C) Neutral tones D) Green tones
- **Answer:** B

---

## 3. Structured Open-Ended (Target: 25-30% total)

### 3.1 Short Description (1-5 words)

**Example 1:**
- **Question:** What color is the woman's shirt?
- **Answer:** light blue

**Example 2:**
- **Question:** What time of day is it?
- **Answer:** early morning

---

### 3.2 List/Enumeration

**Example 1:**
- **Question:** List all the animals visible in the image.
- **Answer:** dog, cat, bird, rabbit

**Example 2:**
- **Question:** What objects are on the table?
- **Answer:** laptop, coffee mug, notebook, pen

---

### 3.3 Detailed Description (sentence-level)

**Example 1:**
- **Question:** Describe the scene in detail.
- **Answer:** A sunny park with children playing on swings while adults watch from nearby benches under shade trees

**Example 2:**
- **Question:** What is the person doing?
- **Answer:** The person is riding a bicycle along a tree-lined path while carrying a backpack

---

### 3.4 Spatial Description

**Example 1:**
- **Question:** Where is the cat relative to the sofa?
- **Answer:** The cat is sitting on the left armrest of the sofa

**Example 2:**
- **Question:** Describe the kitchen layout.
- **Answer:** Kitchen with island in the center, cabinets along the left wall, and stove on the right side

---

## 4. Relational/Comparative (Target: 10-12%)

### 4.1 Size Comparison

**Example 1:**
- **Question:** Which is larger, the red box or the blue box?
- **Answer:** The red box is larger

**Example 2:**
- **Question:** Is the tree taller than the building?
- **Answer:** No, the building is taller than the tree

---

### 4.2 Position Comparison

**Example 1:**
- **Question:** What object is closest to the camera?
- **Answer:** the red car

**Example 2:**
- **Question:** Which person is furthest from the door?
- **Answer:** the person in the yellow jacket

---

### 4.3 Attribute Comparison

**Example 1:**
- **Question:** Which person is taller?
- **Answer:** the person on the left

**Example 2:**
- **Question:** Which dog has darker fur?
- **Answer:** the dog sitting near the tree

---

### 4.4 Temporal Ordering

**Example 1:**
- **Question:** In the sequence of images, what happens first?
- **Answer:** the door opens

**Example 2:**
- **Question:** Which action occurs before the person sits down?
- **Answer:** the person removes their coat

---

## 5. Classification/Categorization (Target: 5-8%)

### 5.1 Object Category

**Example 1:**
- **Question:** What type of vehicle is this?
- **Answer:** sedan

**Example 2:**
- **Question:** What breed of dog is shown?
- **Answer:** golden retriever

---

### 5.2 Scene Type

**Example 1:**
- **Question:** What kind of location is this?
- **Answer:** residential neighborhood

**Example 2:**
- **Question:** What type of environment is depicted?
- **Answer:** urban downtown area

---

### 5.3 Activity Type

**Example 1:**
- **Question:** What sport is being played?
- **Answer:** basketball

**Example 2:**
- **Question:** What type of work is the person performing?
- **Answer:** construction work

---

## 6. Existence/Verification (Target: 25-30%)

### 6.1 Binary Yes/No - Object Existence

**Example 1:**
- **Question:** Is there a dog in the image?
- **Answer:** yes

**Example 2:**
- **Question:** Can you see any birds in the sky?
- **Answer:** no

---

### 6.2 Binary Yes/No - Relationship/Contact

**Example 1:**
- **Question:** Are the two people touching?
- **Answer:** no

**Example 2:**
- **Question:** Is there direct contact between the cup and the table?
- **Answer:** yes

---

### 6.3 Binary Yes/No - Attribute Verification

**Example 1:**
- **Question:** Is the car red?
- **Answer:** yes

**Example 2:**
- **Question:** Is the sky cloudy?
- **Answer:** no

---

## 7. Unanswerable/Insufficient Context (Target: 7-10%)

### 7.1 Hidden Information

**Example 1:**
- **Question:** What is written on the back of the sign?
- **Answer:** Cannot be determined from this angle

**Example 2:**
- **Question:** What brand is the laptop?
- **Answer:** The logo is not visible in the image

---

### 7.2 Non-Visible Information

**Example 1:**
- **Question:** How many rooms are inside the building?
- **Answer:** Not visible in the image

**Example 2:**
- **Question:** What color are the person's shoes?
- **Answer:** The feet are not visible in the frame

---

## 8. Multi-Step Reasoning (Target: 5-8%)

### 8.1 Chain of Reasoning

**Example 1:**
- **Question:** If each person needs 2 chairs and there are 4 people, how many chairs are needed?
- **Answer:** 8 chairs

**Example 2:**
- **Question:** If the clock shows 3:00 and the meeting starts in 30 minutes, what time does the meeting start?
- **Answer:** 3:30

---

### 8.2 Conditional Reasoning

**Example 1:**
- **Question:** Would the cat fit through the cat door shown in the image?
- **Answer:** Yes, the door is large enough for the cat

**Example 2:**
- **Question:** Can the truck pass under the bridge based on their relative heights?
- **Answer:** No, the truck is too tall for the bridge clearance

---

### 8.3 Inference-Based Reasoning

**Example 1:**
- **Question:** Based on the shadows, is the sun high or low in the sky?
- **Answer:** The sun is low based on the long shadows

**Example 2:**
- **Question:** Given the wet pavement, has it recently rained?
- **Answer:** Yes, it appears to have rained recently

---

## 9. False Premises (Hallucination Detection - Maintain Current)

### 9.1 Non-Existent Objects

**Example 1:**
- **Question:** How many sharks are present in the travel brochure?
- **Answer:** There are no sharks in the brochure

**Example 2:**
- **Question:** What color is the elephant in the image?
- **Answer:** There is no elephant in the image

---

### 9.2 Incorrect Assumptions

**Example 1:**
- **Question:** What kind of fruit is the animal holding with its tail?
- **Answer:** The animal is not holding any fruit with its tail

**Example 2:**
- **Question:** How many people are sinking in the boat in the background?
- **Answer:** There is no boat sinking in the background

---

## 10. Visual Challenge Questions (Complex Reasoning - Maintain Current)

### 10.1 Fine-Grained Details

**Example 1:**
- **Question:** How many fingers does the person have including the thumb?
- **Answer:** The person has six fingers

**Example 2:**
- **Question:** How many girls are in the hourglass?
- **Answer:** One girl is in the hourglass

---

### 10.2 Subtle Distinctions

**Example 1:**
- **Question:** Which of the two girls has a necklace on?
- **Answer:** The girl on the right

**Example 2:**
- **Question:** How many children have dark brown hair?
- **Answer:** Two children have dark brown hair

---

## Target Distribution Summary

| Answer Type                          | Current % | Target %  |
|--------------------------------------|-----------|-----------|
| Yes/No (Binary)                      | 65.68%    | 25-30%    |
| Exact Count/Number                   | 6.57%     | 10-12%    |
| Multiple Choice/Selection            | 0.36%     | 10-15%    |
| Short Description (1-5 words)        | 20.08%    | 12-15%    |
| List/Enumeration                     | 0.00%     | 5-8%      |
| Detailed Description (sentence)      | 0.00%     | 8-10%     |
| Spatial Description                  | 0.00%     | 5-7%      |
| Comparative/Relational              | Implicit  | 8-10%     |
| Classification                       | 0.00%     | 5-8%      |
| Unanswerable                         | 7.31%     | 7-10%     |
| Multi-Step Reasoning                 | 0.00%     | 5-8%      |

---

## Implementation Notes

### For Existing Images:
1. Review current yes/no questions - convert ~40% to other types
2. Add counting variants for images with multiple objects
3. Create multiple-choice versions of classification questions
4. Expand open-ended questions with more specific spatial/descriptive prompts

### For Maintaining Hallucination Detection:
- Keep false premise questions (critical for hallucination detection)
- Maintain visual challenge questions (complex reasoning)
- Ensure unanswerable questions remain in dataset
- Balance between detection capabilities and answer diversity

### Quality Assurance:
- Verify ground truth answers are accurate for all types
- Ensure multiple-choice options are plausible but distinguishable
- Validate that unanswerable questions truly cannot be answered from image
- Check that multi-step reasoning follows logical chains

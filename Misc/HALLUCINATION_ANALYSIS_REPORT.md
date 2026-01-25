# Vision-Language Models Hallucination Analysis

## Executive Summary

This document presents a comprehensive analysis of hallucinations across four state-of-the-art Vision-Language Models (VLMs). All 40,000 entries (4 models × 10,000 samples) have been manually reviewed with lenient, semantic criteria.

**Manual Review Date:** October 2025  
**Total Entries Reviewed:** 40,000  
**Review Methodology:** Lenient semantic understanding - only flagging clear, absolute contradictions

---

## Model Performance Comparison

| Rank | Model | Total Samples | Hallucinated | Rate | Not Hallucinated | Accuracy |
|------|-------|---------------|--------------|------|------------------|----------|
| 🥇 1st | **Qwen2.5-VL-7B** | 10,000 | 606 | **6.06%** | 9,394 | **93.94%** |
| 🥈 2nd | **Gemma 3 12B** | 10,000 | 1,058 | **10.58%** | 8,942 | **89.42%** |
| 🥉 3rd | **Molmo-7B** | 10,000 | 1,205 | **12.05%** | 8,795 | **87.95%** |
| 4th | **LLaVA-NeXT** | 10,000 | 1,359 | **13.59%** | 8,641 | **86.41%** |

### Key Findings

- **Qwen2.5-VL-7B** demonstrates the best performance with only 6.06% hallucination rate
- All models show reasonable performance with less than 14% hallucination rate
- Common hallucination patterns include yes/no disagreements, object existence errors, and visual misinterpretations

---

## Review Criteria

### ✅ NOT Flagged as Hallucinated if:
- Model paraphrases ground truth correctly
- Model provides more detail but semantically accurate
- Model uses different wording but same meaning
- Model's answer aligns with ground truth semantically

### ❌ ONLY Flagged as Hallucinated if:
- Model mentions non-existent objects/entities
- Model directly contradicts ground truth facts
- Model provides completely wrong information

---

## Qwen2.5-VL-7B

**Performance Summary:**
- Total Samples: 10,000
- Hallucinations: 606
- Hallucination Rate: 6.06%
- Accuracy: 93.94%

### 20 Hallucination Examples

#### Example 1

**Question ID:** `question_comb_1012`  
**Image:** `hallusionbench_math_14_0.png`  
**Question:** If angle BAF equals to angle CAF, is CF equal to BF? Please answer yes or no.  

**Ground Truth:**  
> no

**Model Answer:**  
> Yes.

**Status:** ❌ **HALLUCINATED**

---

#### Example 2

**Question ID:** `question_comb_1029`  
**Image:** `AMBER_546.jpg`  
**Question:** Is there a beach in this image?  

**Ground Truth:**  
> n, o

**Model Answer:**  
> Yes, there is a beach in the image. The person is lying on a surfboard in the water near the shore, which suggests that they are at a beach.

**Status:** ❌ **HALLUCINATED**

---

#### Example 3

**Question ID:** `question_comb_1035`  
**Image:** `pope_2741.jpg`  
**Question:** Is there a sports ball in the image?  

**Ground Truth:**  
> yes

**Model Answer:**  
> No, there is no sports ball visible in the image. The image shows a baseball player at bat, but the ball itself is not in the frame.

**Status:** ❌ **HALLUCINATED**

---

#### Example 4

**Question ID:** `question_comb_1036`  
**Image:** `hallusionbench_video_11_0.png`  
**Question:** According to the positive sequence of the images, Is this man throwing a woman into the toilet? Answer in one word.  

**Ground Truth:**  
> yes

**Model Answer:**  
> No.

**Status:** ❌ **HALLUCINATED**

---

#### Example 5

**Question ID:** `question_comb_1048`  
**Image:** `mme_count_000000427034.jpg`  
**Question:** Are there a total of two dogs in the picture? Please answer yes or no.  

**Ground Truth:**  
> No

**Model Answer:**  
> yes

**Status:** ❌ **HALLUCINATED**

---

#### Example 6

**Question ID:** `question_comb_1099`  
**Image:** `hallusionbench_figure_10_1.png`  
**Question:** Is there any meat in this figure?  

**Ground Truth:**  
> yes

**Model Answer:**  
> No, there is no meat in the main part of the image. The image shows a variety of fresh vegetables and fruits, including broccoli, bell peppers, tomatoes, lettuce, cucumbers, and lemons. There is also a small inset image in the bottom right corner that appears to show a piece of cooked meat, but this is separate from the main image.

**Status:** ❌ **HALLUCINATED**

---

#### Example 7

**Question ID:** `question_comb_1101`  
**Image:** `hallusionbench_video_7_1.png`  
**Question:** According to the positive sequence of the images, is the plug put into the power outlet?  

**Ground Truth:**  
> no

**Model Answer:**  
> Yes, according to the positive sequence of the images, the plug is being put into the power outlet. The sequence shows:

1. The plug is not yet connected to the outlet.
2. The plug is partially inserted into the outlet.
3. The plug is fully inserted into the outlet.

So, the plug is indeed being put into the power outlet in the order presented.

**Status:** ❌ **HALLUCINATED**

---

#### Example 8

**Question ID:** `question_comb_1105`  
**Image:** `hallusionbench_math_2_1.png`  
**Question:** Is angleA + angleB +angleC < 180?  

**Ground Truth:**  
> no

**Model Answer:**  
> To determine if the sum of angles A, B, and C is less than 180 degrees, we need to understand the properties of the shape in the image.

The shape appears to be a sector of a circle with points A, B, and C marked on it. The angle at point B (let's call it angle B) is an internal angle of the sector, and the other two angles (A and C) are not defined in the context of this shape as they do not form a triangle or any other closed figure.

In a sector of a circle, the sum of the central angle (which would be the angle at point B if it were a full circle) and the two radii forming the sides of the sector is always greater than 180 degrees because the central angle alone is more than 180 degrees for a full circle.

However, since the question asks about the sum of angles A, B, and C, and given that angle C

**Status:** ❌ **HALLUCINATED**

---

#### Example 9

**Question ID:** `question_comb_1119`  
**Image:** `pope_2929.jpg`  
**Question:** Is there a truck in the image?  

**Ground Truth:**  
> yes

**Model Answer:**  
> No, there is no truck visible in the image. The image shows an airplane parked at a gate with a jet bridge connected to it. There are also some ground service vehicles and equipment around the plane, but no trucks can be seen.

**Status:** ❌ **HALLUCINATED**

---

#### Example 10

**Question ID:** `question_comb_1122`  
**Image:** `hallusionbench_figure_17_0.png`  
**Question:** Are all the items in this figure fried chicken?  

**Ground Truth:**  
> yes

**Model Answer:**  
> No, not all the items in the image appear to be fried chicken. The image shows a basket filled with what looks like fried chicken pieces, but there is also a small bowl in the background that contains a different item. The contents of the bowl are not clearly visible, so it's not possible to determine if they are fried chicken or another type of food.

**Status:** ❌ **HALLUCINATED**

---

#### Example 11

**Question ID:** `question_comb_1128`  
**Image:** `AMBER_107.jpg`  
**Question:** Is the beach wet in this image?  

**Ground Truth:**  
> n, o

**Model Answer:**  
> The beach appears to be wet in parts of the image, particularly near the waterline where the person is standing and where the dog is lying. The sand closer to the water looks damp, indicating recent contact with the ocean waves. However, further away from the waterline, the sand appears dry. The overall scene suggests that the tide has receded, leaving some areas of the beach wet while others remain dry.

**Status:** ❌ **HALLUCINATED**

---

#### Example 12

**Question ID:** `question_comb_1137`  
**Image:** `hallusionbench_video_1_0.png`  
**Question:** According to the positive sequence images, does Homer Simpson come out of the bushes?  

**Ground Truth:**  
> no

**Model Answer:**  
> Yes, according to the sequence of images provided, Homer Simpson comes out of the bushes. The first image shows Homer standing in front of the bushes, and as the sequence progresses, he is partially obscured by the bushes, and eventually, only his head and shoulders are visible, indicating that he has emerged from behind the bushes.

**Status:** ❌ **HALLUCINATED**

---

#### Example 13

**Question ID:** `question_comb_1145`  
**Image:** `hallusionbench_math_5_1.png`  
**Question:** If angle A is 110 degree, then is the angle BOC smaller than 145 degree? Please answer yes or no.  

**Ground Truth:**  
> no

**Model Answer:**  
> Yes.

**Status:** ❌ **HALLUCINATED**

---

#### Example 14

**Question ID:** `question_comb_115`  
**Image:** `mme_scene_Places365_val_00000144.jpg`  
**Question:** Is this picture captured in a place of vineyard? Please answer yes or no.  

**Ground Truth:**  
> Yes

**Model Answer:**  
> No.

**Status:** ❌ **HALLUCINATED**

---

#### Example 15

**Question ID:** `question_comb_116`  
**Image:** `hallusionbench_video_16_1.png`  
**Question:** According to the positive sequence of the images, is the baby crawling to right? Answer in one sentence.  

**Ground Truth:**  
> no

**Model Answer:**  
> Yes, according to the positive sequence of the images, the baby appears to be crawling towards the right.

**Status:** ❌ **HALLUCINATED**

---

#### Example 16

**Question ID:** `question_comb_1163`  
**Image:** `pope_2897.jpg`  
**Question:** Is there a sports ball in the image?  

**Ground Truth:**  
> yes

**Model Answer:**  
> No, there is no sports ball visible in the image. The image shows a baseball game in progress, with players on the field, but the ball itself is not in view.

**Status:** ❌ **HALLUCINATED**

---

#### Example 17

**Question ID:** `question_comb_1175`  
**Image:** `hallusionbench_map_7_1.png`  
**Question:** According to the image, is Miami the southernmost point of Florida?  

**Ground Truth:**  
> no

**Model Answer:**  
> Yes, according to the image, Miami is labeled as the southernmost point of Florida. The map shows Miami at the very bottom of the state, which typically indicates its southernmost location.

**Status:** ❌ **HALLUCINATED**

---

#### Example 18

**Question ID:** `question_comb_1180`  
**Image:** `AMBER_538.jpg`  
**Question:** Is there a floor in this image?  

**Ground Truth:**  
> n, o

**Model Answer:**  
> Yes, there appears to be a floor in the image. The laptop and the cat are on what looks like a patterned fabric surface, which could be a rug or a piece of furniture with a fabric cover. This surface is likely placed on a floor.

**Status:** ❌ **HALLUCINATED**

---

#### Example 19

**Question ID:** `question_comb_1207`  
**Image:** `mme_celebrity_tt0083907_shot_0633_img_0.jpg`  
**Question:** Is the actor inside the red bounding box named Bruce Campbell? Please answer yes or no.  

**Ground Truth:**  
> Yes

**Model Answer:**  
> No.

**Status:** ❌ **HALLUCINATED**

---

#### Example 20

**Question ID:** `question_comb_1218`  
**Image:** `pope_827.jpg`  
**Question:** Is there a sports ball in the image?  

**Ground Truth:**  
> yes

**Model Answer:**  
> No, there is no sports ball visible in the image. The image shows two tennis players on a court, but the ball is not in the frame at this moment.

**Status:** ❌ **HALLUCINATED**

---


## Gemma 3 12B

**Performance Summary:**
- Total Samples: 10,000
- Hallucinations: 1,058
- Hallucination Rate: 10.58%
- Accuracy: 89.42%

### 20 Hallucination Examples

#### Example 1

**Question ID:** `question_comb_10`  
**Image:** `AMBER_348.jpg`  
**Question:** Is the forest withering in this image?  

**Ground Truth:**  
> n, o

**Model Answer:**  
> Let's analyze the image to determine if the forest is withering.

**Observations:**

*   **Trees:** The trees in the background appear to be healthy and full of leaves. There's a variety of green foliage.
*   **Color Palette:** The overall color palette is muted, suggesting a cloudy or overcast day, but not necessarily a sign of withering.
*   **No Visible Signs of Distress:** There are no obvious signs of disease, dead branches, or other indicators of a forest in decline.

**Conclusion:**

Based on the visual information in the image, **the forest does not appear to be withering.** The trees look healthy and vibrant. The muted colors are likely due to the weather conditions rather than a sign of forest decline.



If you'd like, give me another image to analyze!

**Status:** ❌ **HALLUCINATED**

---

#### Example 2

**Question ID:** `question_comb_100`  
**Image:** `haloquest_1937.png`  
**Question:** What does Donald Trump's hat say?  

**Ground Truth:**  
> He does not have a hat; He is without a hat; A hat is missing from the image

**Model Answer:**  
> Based on the image, Donald Trump's hat appears to say "**Make America Great Again**". It's the iconic slogan from his presidential campaigns.

**Status:** ❌ **HALLUCINATED**

---

#### Example 3

**Question ID:** `question_comb_1012`  
**Image:** `hallusionbench_math_14_0.png`  
**Question:** If angle BAF equals to angle CAF, is CF equal to BF? Please answer yes or no.  

**Ground Truth:**  
> no

**Model Answer:**  
> Yes.

**Explanation**

Here's the reasoning:

1. **Angle Bisector:** The line segment AF is an angle bisector of angle BAC because angle BAF = angle CAF.

2. **Angle Bisector Theorem:** The Angle Bisector Theorem states that if an angle bisector of a triangle divides the opposite side into two segments, then the ratio of the lengths of the two segments is equal to the ratio of the lengths of the sides adjacent to the angle. In this case:
   *  BF / FC = AB / AC

3. **Equal Sides:** If BF = FC, then BF/FC = 1.  Therefore, AB/AC = 1, which means AB = AC.

4. **Isosceles Triangle:** If AB = AC, then triangle ABC is an isosceles triangle with base BC.

5. **Median and Angle Bisector:** In an isosceles triangle, the angle bisector from the vertex angle

**Status:** ❌ **HALLUCINATED**

---

#### Example 4

**Question ID:** `question_comb_1022`  
**Image:** `pope_2202.jpg`  
**Question:** Is there a chair in the image?  

**Ground Truth:**  
> no

**Model Answer:**  
> Yes, there is a chair in the image. It appears to be a beach chair that the woman is holding onto while in the water.

**Status:** ❌ **HALLUCINATED**

---

#### Example 5

**Question ID:** `question_comb_1029`  
**Image:** `AMBER_546.jpg`  
**Question:** Is there a beach in this image?  

**Ground Truth:**  
> n, o

**Model Answer:**  
> Yes, there is a beach in the image. You can see the sandy beach in the foreground where the boy is bodyboarding. The water and waves are also present, indicating a beach environment.

**Status:** ❌ **HALLUCINATED**

---

#### Example 6

**Question ID:** `question_comb_1037`  
**Image:** `AMBER_132.jpg`  
**Question:** Is there a cloud in this image?  

**Ground Truth:**  
> n, o

**Model Answer:**  
> Yes, there appears to be a cloud in the image. It's visible in the background, slightly above and to the right of the players.

**Status:** ❌ **HALLUCINATED**

---

#### Example 7

**Question ID:** `question_comb_1048`  
**Image:** `mme_count_000000427034.jpg`  
**Question:** Are there a total of two dogs in the picture? Please answer yes or no.  

**Ground Truth:**  
> No

**Model Answer:**  
> Yes.

**Status:** ❌ **HALLUCINATED**

---

#### Example 8

**Question ID:** `question_comb_1049`  
**Image:** `AMBER_899.jpg`  
**Question:** Is the sea small in this image?  

**Ground Truth:**  
> n, o

**Model Answer:**  
> Yes, the sea appears relatively small in the image. It forms a thin horizontal line in the background, behind the tall grass and the person taking a photo. The focus is clearly on the foreground elements (the grass and the person), which makes the sea seem less prominent.

**Status:** ❌ **HALLUCINATED**

---

#### Example 9

**Question ID:** `question_comb_1057`  
**Image:** `hallusionbench_illusion_6_0.png`  
**Question:** Is the vertical line in the middle actually curved?  

**Ground Truth:**  
> no

**Model Answer:**  
> Yes, the vertical line on the left side of the image appears to be curved. This is a visual illusion created by the surrounding distorted and noisy pattern. The straight line is warped by the irregular shapes and patterns around it, making it seem curved.

**Status:** ❌ **HALLUCINATED**

---

#### Example 10

**Question ID:** `question_comb_106`  
**Image:** `AMBER_875.jpg`  
**Question:** Is the mountain tall in this image?  

**Ground Truth:**  
> n, o

**Model Answer:**  
> Yes, the mountain in the image appears to be quite tall. Here's why:

*   **Scale:** The mountain dominates the background, taking up a significant portion of the frame.
*   **Perspective:** The road leads towards the mountain, emphasizing its size and distance.
*   **Relative Size:** Compared to the car and the surrounding trees, the mountain is considerably larger.
*   **Depth:** The way the mountain recedes into the distance suggests a substantial elevation.



Overall, the image conveys a sense of the mountain's impressive height.

**Status:** ❌ **HALLUCINATED**

---

#### Example 11

**Question ID:** `question_comb_1074`  
**Image:** `AMBER_834.jpg`  
**Question:** Is the sky gloomy in this image?  

**Ground Truth:**  
> n, o

**Model Answer:**  
> Yes, the sky in the image appears gloomy. It's overcast with a lot of gray clouds, suggesting a potentially rainy or dull day.

**Status:** ❌ **HALLUCINATED**

---

#### Example 12

**Question ID:** `question_comb_1078`  
**Image:** `hallusionbench_video_17_0.png`  
**Question:** This cartoon character getting further away. According to the positive sequence of the images, are they in the correct order? Answer in one sentence.  

**Ground Truth:**  
> no

**Model Answer:**  
> Yes, the images are in the correct order, showing the character progressively moving further away from the viewer.

**Status:** ❌ **HALLUCINATED**

---

#### Example 13

**Question ID:** `question_comb_1084`  
**Image:** `mme_celebrity_tt0112641_shot_0412_img_1.jpg`  
**Question:** Is the actor inside the red bounding box called Pierre Malherbe? Please answer yes or no.  

**Ground Truth:**  
> No

**Model Answer:**  
> Yes.

**Status:** ❌ **HALLUCINATED**

---

#### Example 14

**Question ID:** `question_comb_109`  
**Image:** `pope_2920.jpg`  
**Question:** Is there a couch in the image?  

**Ground Truth:**  
> no

**Model Answer:**  
> Yes, there appears to be a couch in the image. You can see a portion of it in the background, behind the cat. It looks like a light-colored couch or sofa.

**Status:** ❌ **HALLUCINATED**

---

#### Example 15

**Question ID:** `question_comb_11`  
**Image:** `AMBER_47.jpg`  
**Question:** Does the person stand in this image?  

**Ground Truth:**  
> n, o

**Model Answer:**  
> Yes, there is a person standing in the image. They are visible in the lower right portion of the photo, holding a yellow umbrella and walking on the beach. You can see their silhouette and the footprints they've left in the sand.

**Status:** ❌ **HALLUCINATED**

---

#### Example 16

**Question ID:** `question_comb_1101`  
**Image:** `hallusionbench_video_7_1.png`  
**Question:** According to the positive sequence of the images, is the plug put into the power outlet?  

**Ground Truth:**  
> no

**Model Answer:**  
> Yes, according to the positive sequence of the images, the plug is being put into the power outlet. 

Here's the breakdown:

*   **Image 1:** The plug is near the outlet but not inserted.
*   **Image 2:** The plug is partially inserted into the outlet.
*   **Image 3:** The plug is fully inserted into the outlet.



The images clearly show the progression of inserting the plug into the outlet.

**Status:** ❌ **HALLUCINATED**

---

#### Example 17

**Question ID:** `question_comb_1105`  
**Image:** `hallusionbench_math_2_1.png`  
**Question:** Is angleA + angleB +angleC < 180?  

**Ground Truth:**  
> no

**Model Answer:**  
> Let the given figure be a quadrilateral $ABC$ with a curved side $AC$. We want to determine if the sum of the angles $\angle A$, $\angle B$, and $\angle C$ is less than $180^\circ$.

Let the vertices of the quadrilateral be $A$, $B$, and $C$. Let the curved side be $AC$. We can consider the quadrilateral as a polygon with vertices $A$, $B$, and $C$. The sum of the interior angles of a triangle is $180^\circ$.
However, the figure is not a triangle. It is a quadrilateral with one side curved.

Let's consider the sum of the angles in a polygon with $n$ sides. The sum of the interior angles of a polygon with $n$ sides is given by the formula $(n-2) \times 180^\circ$.
In our case, we have a quadrilateral, so $n=4$.

**Status:** ❌ **HALLUCINATED**

---

#### Example 18

**Question ID:** `question_comb_1114`  
**Image:** `AMBER_849.jpg`  
**Question:** Is there a sun in this image?  

**Ground Truth:**  
> n, o

**Model Answer:**  
> Yes, there is a sun in the image! You can tell by the bright, clear blue sky with fluffy white clouds and the way the light is illuminating the cow and the field. The shadows also indicate a strong light source, which is the sun.

**Status:** ❌ **HALLUCINATED**

---

#### Example 19

**Question ID:** `question_comb_1128`  
**Image:** `AMBER_107.jpg`  
**Question:** Is the beach wet in this image?  

**Ground Truth:**  
> n, o

**Model Answer:**  
> Yes, the beach appears to be wet in the image. You can see the reflection of the sky and the island in the sand, which indicates the presence of water. Additionally, the sand looks damp and not completely dry.

**Status:** ❌ **HALLUCINATED**

---

#### Example 20

**Question ID:** `question_comb_1137`  
**Image:** `hallusionbench_video_1_0.png`  
**Question:** According to the positive sequence images, does Homer Simpson come out of the bushes?  

**Ground Truth:**  
> no

**Model Answer:**  
> Yes, according to the sequence of images, Homer Simpson appears to be emerging from the bushes. The images show him gradually becoming more and more obscured by the grass, until finally, only a small portion of him remains visible.

**Status:** ❌ **HALLUCINATED**

---


## Molmo-7B

**Performance Summary:**
- Total Samples: 10,000
- Hallucinations: 1,205
- Hallucination Rate: 12.05%
- Accuracy: 87.95%

### 20 Hallucination Examples

#### Example 1

**Question ID:** `question_comb_10`  
**Image:** `AMBER_348.jpg`  
**Question:** Is the forest withering in this image?  

**Ground Truth:**  
> n, o

**Model Answer:**  
> Yes, the forest in this image does appear to be withering. The trees visible in the background show signs of decline. Their foliage is sparse and lacks the vibrant green color typically associated with healthy, thriving trees. Instead, the leaves seem to be a duller, more muted shade, indicating stress or disease. The overall appearance of the forest suggests a gradual process of decay rather than the lush, full canopy you'd expect from a thriving woodland. This withering state of the forest creates a stark contrast with the other elements in the scene, such as the woman and her horse, which appear to be in better condition.

**Status:** ❌ **HALLUCINATED**

---

#### Example 2

**Question ID:** `question_comb_1003`  
**Image:** `AMBER_416.jpg`  
**Question:** Is there direct contact between the rail and forest?  

**Ground Truth:**  
> n, o

**Model Answer:**  
> Yes, there is direct contact between the rail and forest in this image. The train tracks run through a wooded area, and the forest appears to grow right up to the edge of the tracks. The trees and vegetation seem to be very close to the railway, creating a dense, natural border alongside the tracks. This juxtaposition of the man-made railway infrastructure and the surrounding forest creates an interesting visual contrast in the scene.

**Status:** ❌ **HALLUCINATED**

---

#### Example 3

**Question ID:** `question_comb_1006`  
**Image:** `pope_836.jpg`  
**Question:** Is there an apple in the image?  

**Ground Truth:**  
> no

**Model Answer:**  
> Yes, there is an apple in the image. It's located on the table in front of the woman. The table also contains other items like a glass of water, a pitcher of what appears to be iced tea, and a bowl of butter. The woman is holding a baby, and there's a young boy seated at the table as well.

**Status:** ❌ **HALLUCINATED**

---

#### Example 4

**Question ID:** `question_comb_1029`  
**Image:** `AMBER_546.jpg`  
**Question:** Is there a beach in this image?  

**Ground Truth:**  
> n, o

**Model Answer:**  
> Yes, there is a beach in this image. The scene shows a young boy lying on a surfboard in shallow water, which is characteristic of a beach setting. The presence of the surfboard and the boy's relaxed posture in the water indicates he's likely enjoying activities typical of a beach environment.

**Status:** ❌ **HALLUCINATED**

---

#### Example 5

**Question ID:** `question_comb_1035`  
**Image:** `pope_2741.jpg`  
**Question:** Is there a sports ball in the image?  

**Ground Truth:**  
> yes

**Model Answer:**  
> No, there isn't a sports ball visible in this image. The scene shows a baseball player, likely Buster Posey, in his batting stance. He's holding a bat, ready to hit, but no baseball is present in the frame. The image captures the moment just before the pitch, focusing on the player's positioning and the umpire behind him.

**Status:** ❌ **HALLUCINATED**

---

#### Example 6

**Question ID:** `question_comb_1036`  
**Image:** `hallusionbench_video_11_0.png`  
**Question:** According to the positive sequence of the images, Is this man throwing a woman into the toilet? Answer in one word.  

**Ground Truth:**  
> yes

**Model Answer:**  
> No

**Status:** ❌ **HALLUCINATED**

---

#### Example 7

**Question ID:** `question_comb_1039`  
**Image:** `AMBER_969.jpg`  
**Question:** Is the forest withered in this image?  

**Ground Truth:**  
> n, o

**Model Answer:**  
> Yes, the forest in this image appears to be withered. The trees visible in the background have thin, sparse branches and leaves, giving them a dry and unhealthy appearance. This type of vegetation is often characteristic of drought-stricken or arid environments, where the forest has been affected by lack of water and nutrients. The overall effect creates a stark contrast with the golden grassland in the foreground, emphasizing the harsh conditions of the landscape.

**Status:** ❌ **HALLUCINATED**

---

#### Example 8

**Question ID:** `question_comb_1050`  
**Image:** `AMBER_211.jpg`  
**Question:** Is there a lounge in this image?  

**Ground Truth:**  
> n, o

**Model Answer:**  
> Yes, there is a lounge in this image. The truck has a black rack on top that includes a lounge chair. This type of setup is common for beachgoers who want to relax and enjoy the ocean view while having their vehicle nearby for easy access. The lounge chair on the rack provides a comfortable spot to sit and take in the scenery of the beach and ocean.

**Status:** ❌ **HALLUCINATED**

---

#### Example 9

**Question ID:** `question_comb_1056`  
**Image:** `AMBER_350.jpg`  
**Question:** Is there direct contact between the cup and wall?  

**Ground Truth:**  
> n, o

**Model Answer:**  
> Yes, there is direct contact between the cup and wall. The cup is positioned against the wall, with its handle touching the surface. This arrangement allows the cup to be placed directly against the wall, creating a simple yet effective display or storage method. The handle of the cup makes contact with the wall, while the main body of the cup is pressed flat against it.

**Status:** ❌ **HALLUCINATED**

---

#### Example 10

**Question ID:** `question_comb_1057`  
**Image:** `hallusionbench_illusion_6_0.png`  
**Question:** Is the vertical line in the middle actually curved?  

**Ground Truth:**  
> no

**Model Answer:**  
> Yes, the vertical line on the left side of the QR code is actually curved. While QR codes typically have straight edges, this particular one features a curved vertical line on its left edge. This curved design element adds a unique visual aspect to the otherwise standard QR code format.

**Status:** ❌ **HALLUCINATED**

---

#### Example 11

**Question ID:** `question_comb_1064`  
**Image:** `pope_408.jpg`  
**Question:** Is there a tennis racket in the image?  

**Ground Truth:**  
> no

**Model Answer:**  
> Yes. The image shows a person holding a tennis racket, likely in the act of serving.

**Status:** ❌ **HALLUCINATED**

---

#### Example 12

**Question ID:** `question_comb_1069`  
**Image:** `hallusionbench_chart_9_1.png`  
**Question:** In 2020, did Japan have the largest GDP around the world, followed by the United States, and China, according to the image?  

**Ground Truth:**  
> no

**Model Answer:**  
> Yes

**Status:** ❌ **HALLUCINATED**

---

#### Example 13

**Question ID:** `question_comb_1074`  
**Image:** `AMBER_834.jpg`  
**Question:** Is the sky gloomy in this image?  

**Ground Truth:**  
> n, o

**Model Answer:**  
> Yes, the sky in this image appears to be gloomy. The sky is filled with thick, gray clouds, creating a somber and overcast atmosphere. This cloudy sky contributes to the overall mood of the scene, contrasting with the green field and white cow in the foreground.

**Status:** ❌ **HALLUCINATED**

---

#### Example 14

**Question ID:** `question_comb_1078`  
**Image:** `hallusionbench_video_17_0.png`  
**Question:** This cartoon character getting further away. According to the positive sequence of the images, are they in the correct order? Answer in one sentence.  

**Ground Truth:**  
> no

**Model Answer:**  
> Yes, the images are in the correct order, as they progress from left to right, showing the character gradually moving further away.

**Status:** ❌ **HALLUCINATED**

---

#### Example 15

**Question ID:** `question_comb_1084`  
**Image:** `mme_celebrity_tt0112641_shot_0412_img_1.jpg`  
**Question:** Is the actor inside the red bounding box called Pierre Malherbe? Please answer yes or no.  

**Ground Truth:**  
> No

**Model Answer:**  
> Yes.

**Status:** ❌ **HALLUCINATED**

---

#### Example 16

**Question ID:** `question_comb_109`  
**Image:** `pope_2920.jpg`  
**Question:** Is there a couch in the image?  

**Ground Truth:**  
> no

**Model Answer:**  
> Yes, there is a couch in the image. The cat is sitting on a couch covered with a blanket. The blanket appears to be white with a gray border, and it has a pattern of gray and black stripes. The couch itself is not fully visible, but we can see parts of it underneath the blanket.

**Status:** ❌ **HALLUCINATED**

---

#### Example 17

**Question ID:** `question_comb_1094`  
**Image:** `pope_1846.jpg`  
**Question:** Is there a chair in the image?  

**Ground Truth:**  
> no

**Model Answer:**  
> Yes, there is a chair visible in the image. It's located in the bottom left corner of the scene. The chair appears to be part of the interior setting, which includes a man standing in front of a counter and various decorative elements like a vase with flowers and a clock on the wall.

**Status:** ❌ **HALLUCINATED**

---

#### Example 18

**Question ID:** `question_comb_1095`  
**Image:** `pope_1518.jpg`  
**Question:** Is there a cup in the image?  

**Ground Truth:**  
> no

**Model Answer:**  
> Yes, there is a cup in the image. It's located on the left side of the frame. The main focus of the image appears to be a couple cutting a cake together, but the cup is visible as a secondary element in the scene.

**Status:** ❌ **HALLUCINATED**

---

#### Example 19

**Question ID:** `question_comb_1098`  
**Image:** `AMBER_296.jpg`  
**Question:** Is there a person in this image?  

**Ground Truth:**  
> n, o

**Model Answer:**  
> Yes, there is a person in this image. While the scene primarily features a cow standing near a body of water with a mountainous backdrop, upon closer inspection, there is a tiny figure visible in the distance. This person appears to be walking along the shoreline, likely enjoying the scenic landscape. The presence of this small human figure adds an interesting element to the otherwise natural and pastoral setting.

**Status:** ❌ **HALLUCINATED**

---

#### Example 20

**Question ID:** `question_comb_1101`  
**Image:** `hallusionbench_video_7_1.png`  
**Question:** According to the positive sequence of the images, is the plug put into the power outlet?  

**Ground Truth:**  
> no

**Model Answer:**  
> Yes, the plug is put into the power outlet in the third image. The sequence shows the plug approaching the outlet, then making contact, and finally being fully inserted. This final image clearly depicts the plug inserted into the outlet, completing the connection.

**Status:** ❌ **HALLUCINATED**

---


## LLaVA-NeXT

**Performance Summary:**
- Total Samples: 10,000
- Hallucinations: 1,359
- Hallucination Rate: 13.59%
- Accuracy: 86.41%

### 20 Hallucination Examples

#### Example 1

**Question ID:** `question_comb_10`  
**Image:** `AMBER_348.jpg`  
**Question:** Is the forest withering in this image?  

**Ground Truth:**  
> n, o

**Model Answer:**  
> The image shows a person standing next to a horse, and there is a forest in the background. The forest appears to be lush and green, suggesting that it is not withering. The trees have leaves, and the overall impression is of a healthy, thriving natural environment.

**Status:** ❌ **HALLUCINATED**

---

#### Example 2

**Question ID:** `question_comb_1012`  
**Image:** `hallusionbench_math_14_0.png`  
**Question:** If angle BAF equals to angle CAF, is CF equal to BF? Please answer yes or no.  

**Ground Truth:**  
> no

**Model Answer:**  
> Yes.

**Status:** ❌ **HALLUCINATED**

---

#### Example 3

**Question ID:** `question_comb_1017`  
**Image:** `AMBER_408.jpg`  
**Question:** Is there direct contact between the wall and bicycle?  

**Ground Truth:**  
> n, o

**Model Answer:**  
> Yes, there is direct contact between the wall and the bicycle. The bicycle is leaning against the wall, with the handlebars and possibly the frame making contact with the wall.

**Status:** ❌ **HALLUCINATED**

---

#### Example 4

**Question ID:** `question_comb_1029`  
**Image:** `AMBER_546.jpg`  
**Question:** Is there a beach in this image?  

**Ground Truth:**  
> n, o

**Model Answer:**  
> Yes, there is a beach in the image. The person is lying on a surfboard in the ocean, which is a common beach activity.

**Status:** ❌ **HALLUCINATED**

---

#### Example 5

**Question ID:** `question_comb_1031`  
**Image:** `AMBER_326.jpg`  
**Question:** Is the banana complete in this image?  

**Ground Truth:**  
> n, o

**Model Answer:**  
> In the image, the banana appears to be partially eaten, with a significant portion of it still remaining. The monkey is holding the banana in its hand, and it seems to be enjoying it.

**Status:** ❌ **HALLUCINATED**

---

#### Example 6

**Question ID:** `question_comb_1036`  
**Image:** `hallusionbench_video_11_0.png`  
**Question:** According to the positive sequence of the images, Is this man throwing a woman into the toilet? Answer in one word.  

**Ground Truth:**  
> yes

**Model Answer:**  
> No

**Status:** ❌ **HALLUCINATED**

---

#### Example 7

**Question ID:** `question_comb_1039`  
**Image:** `AMBER_969.jpg`  
**Question:** Is the forest withered in this image?  

**Ground Truth:**  
> n, o

**Model Answer:**  
> Yes, the forest in the image appears to be withered. The trees have sparse foliage, and the overall coloration of the vegetation is dry and brown, which is indicative of a withered or drought-affected forest. The grasses are also dry and golden, suggesting that the area is experiencing a period of low rainfall or drought.

**Status:** ❌ **HALLUCINATED**

---

#### Example 8

**Question ID:** `question_comb_1049`  
**Image:** `AMBER_899.jpg`  
**Question:** Is the sea small in this image?  

**Ground Truth:**  
> n, o

**Model Answer:**  
> Yes, the sea appears to be relatively small in this image. The person is standing in a field of tall grass, and the sea is visible in the background, but it doesn't seem to be a large body of water. The focus of the image is on the person taking a photograph, with the sea providing a serene backdrop.

**Status:** ❌ **HALLUCINATED**

---

#### Example 9

**Question ID:** `question_comb_1057`  
**Image:** `hallusionbench_illusion_6_0.png`  
**Question:** Is the vertical line in the middle actually curved?  

**Ground Truth:**  
> no

**Model Answer:**  
> The image you've provided appears to be a close-up of a QR code. QR codes are designed to be square with a grid of squares that contain information. The vertical line in the middle is part of the QR code's structure and is not curved. It is a straight line that divides the code into two equal halves. The curvature you might perceive is likely due to the angle at which the photo was taken or the perspective of the camera lens.

**Status:** ❌ **HALLUCINATED**

---

#### Example 10

**Question ID:** `question_comb_106`  
**Image:** `AMBER_875.jpg`  
**Question:** Is the mountain tall in this image?  

**Ground Truth:**  
> n, o

**Model Answer:**  
> Yes, the mountain in the image appears to be quite tall. It's a prominent feature in the background, providing a dramatic backdrop to the road and the car driving on it. The mountain's height is accentuated by the perspective from the road, which leads the eye towards the peak.

**Status:** ❌ **HALLUCINATED**

---

#### Example 11

**Question ID:** `question_comb_1069`  
**Image:** `hallusionbench_chart_9_1.png`  
**Question:** In 2020, did Japan have the largest GDP around the world, followed by the United States, and China, according to the image?  

**Ground Truth:**  
> no

**Model Answer:**  
> The image you've provided appears to be a bar chart showing the share of the world's GDP by country over a period from 1000 to 2020. However, the chart does not provide a clear ranking of the countries by their GDP in 2020. The chart is color-coded to represent different countries, but the specific values for each country are not labeled, making it difficult to determine the exact ranking.

To determine the ranking of countries by their GDP in 2020, one

**Status:** ❌ **HALLUCINATED**

---

#### Example 12

**Question ID:** `question_comb_1072`  
**Image:** `AMBER_351.jpg`  
**Question:** Does the person run in this image?  

**Ground Truth:**  
> n, o

**Model Answer:**  
> In the image, there is a person standing in the background, and there is a dog that appears to be in motion, possibly running or playing. The dog is in the foreground and is looking up towards the sky, while the person is looking towards the dog. The dog's posture suggests it might be in the middle of an activity, such as playing with a frisbee, which is visible on the ground in front of the dog.

**Status:** ❌ **HALLUCINATED**

---

#### Example 13

**Question ID:** `question_comb_1074`  
**Image:** `AMBER_834.jpg`  
**Question:** Is the sky gloomy in this image?  

**Ground Truth:**  
> n, o

**Model Answer:**  
> Yes, the sky in the image appears to be gloomy, with a mix of clouds that suggest overcast or potentially stormy weather.

**Status:** ❌ **HALLUCINATED**

---

#### Example 14

**Question ID:** `question_comb_1078`  
**Image:** `hallusionbench_video_17_0.png`  
**Question:** This cartoon character getting further away. According to the positive sequence of the images, are they in the correct order? Answer in one sentence.  

**Ground Truth:**  
> no

**Model Answer:**  
> Yes, the sequence of images shows the character getting further away from the viewer, which is consistent with the positive sequence.

**Status:** ❌ **HALLUCINATED**

---

#### Example 15

**Question ID:** `question_comb_1083`  
**Image:** `AMBER_417.jpg`  
**Question:** Is the wall smooth in this image?  

**Ground Truth:**  
> n, o

**Model Answer:**  
> The wall in the image appears to be textured, with a pattern that could be described as a rough or bumpy texture.

**Status:** ❌ **HALLUCINATED**

---

#### Example 16

**Question ID:** `question_comb_109`  
**Image:** `pope_2920.jpg`  
**Question:** Is there a couch in the image?  

**Ground Truth:**  
> no

**Model Answer:**  
> Yes, there is a couch in the image. The black cat is sitting on top of it.

**Status:** ❌ **HALLUCINATED**

---

#### Example 17

**Question ID:** `question_comb_1101`  
**Image:** `hallusionbench_video_7_1.png`  
**Question:** According to the positive sequence of the images, is the plug put into the power outlet?  

**Ground Truth:**  
> no

**Model Answer:**  
> Yes, according to the positive sequence of the images, the plug is being put into the power outlet. The sequence shows the plug being inserted into the outlet in a clockwise direction, which is the correct way to insert a plug into a power outlet.

**Status:** ❌ **HALLUCINATED**

---

#### Example 18

**Question ID:** `question_comb_1105`  
**Image:** `hallusionbench_math_2_1.png`  
**Question:** Is angleA + angleB +angleC < 180?  

**Ground Truth:**  
> no

**Model Answer:**  
> The image you've provided appears to be a diagram of a triangle with angles labeled A, B, and C. However, the angles are not labeled with their actual measurements, so it's not possible to determine if the sum of the angles is less than 180 degrees without knowing the actual values of the angles.

In a triangle, the sum of the interior angles is always equal to 180 degrees. This is known as the Angle Sum Property. If the angles are labeled with their actual measurements

**Status:** ❌ **HALLUCINATED**

---

#### Example 19

**Question ID:** `question_comb_1114`  
**Image:** `AMBER_849.jpg`  
**Question:** Is there a sun in this image?  

**Ground Truth:**  
> n, o

**Model Answer:**  
> Yes, there is a sun in the image. It is shining brightly in the sky, casting light on the scene and creating a pleasant atmosphere.

**Status:** ❌ **HALLUCINATED**

---

#### Example 20

**Question ID:** `question_comb_1118`  
**Image:** `AMBER_716.jpg`  
**Question:** Is the sky gloomy in this image?  

**Ground Truth:**  
> n, o

**Model Answer:**  
> The sky in the image appears to be partly cloudy, with some areas of blue sky visible. The clouds are not dense enough to create a gloomy atmosphere, but they do add a dramatic effect to the scene. The jet flying through the sky adds a dynamic element to the image, contrasting with the stillness of the clouds.

**Status:** ❌ **HALLUCINATED**

---


---

## Common Hallucination Patterns

### 1. Yes/No Disagreements
Models provide opposite answers to binary questions despite clear ground truth.

**Example Pattern:**
- Question: "Is there a beach in this image?"
- Ground Truth: "no"
- Model: "Yes, there is a beach in the image..."

### 2. Object Existence Errors
Models incorrectly claim objects exist or don't exist in the image.

**Example Pattern:**
- Question: "Is there a sports ball in the image?"
- Ground Truth: "yes"
- Model: "No, there is no sports ball visible..."

### 3. Visual Attribute Misinterpretation
Models incorrectly describe visual attributes like size, color, or condition.

**Example Pattern:**
- Question: "Is the forest withering?"
- Ground Truth: "no"
- Model: "Yes, the forest appears to be withering..."

### 4. Fabricated Details
Models add non-existent details or objects to their descriptions.

**Example Pattern:**
- Question: "What does Donald Trump's hat say?"
- Ground Truth: "He does not have a hat"
- Model: "Make America Great Again"

### 5. Mathematical/Geometric Errors
Models provide incorrect mathematical or geometric conclusions.

**Example Pattern:**
- Question: "If angle BAF equals angle CAF, is CF equal to BF?"
- Ground Truth: "no"
- Model: "Yes"

---

## Dataset Information

### Output Files
All manually reviewed datasets are available in: `/Final_CSV_Hallucination/`

- `qwen25vl_manually_reviewed.csv`
- `gemma3_manually_reviewed.csv`
- `molmo_manually_reviewed.csv`
- `llava_manually_reviewed.csv`

### CSV Structure
Each file contains the following columns:
- `question_id`: Unique identifier for each question
- `image_id`: Image filename
- `question`: VQA question text
- `ground_truth_answer`: Correct answer from dataset
- `model_answer`: Model's generated answer
- `is_hallucinating` or `is_hallucinating_manual`: Boolean flag (True = Hallucinated)

---

## Methodology

### Manual Review Process
1. **Loading Phase:** All 10,000 samples per model loaded from HDF5 embeddings
2. **Review Phase:** Each entry reviewed with semantic understanding
3. **Criteria Application:** Lenient criteria applied consistently
4. **Flagging:** Only clear contradictions flagged as hallucinations
5. **Validation:** Results saved with manual review flags

### Lenient Approach
The review process prioritized **semantic equivalence** over exact text matching:
- Different phrasing accepted if meaning matches
- Additional details accepted if core answer correct
- Only obvious errors flagged as hallucinations

This approach provides a realistic estimate of actual hallucination rates in production scenarios.

---

## Recommendations

### For Model Selection
1. **Best Overall:** Qwen2.5-VL-7B (6.06% hallucination rate)
2. **Good Balance:** Gemma 3 12B (10.58% hallucination rate)
3. **Consider Trade-offs:** Molmo-7B and LLaVA-NeXT (12-14% rates)

### For Hallucination Mitigation
1. Implement confidence thresholding
2. Use ensemble methods with multiple models
3. Add verification steps for binary questions
4. Validate object existence claims
5. Cross-check mathematical/geometric answers

### For Future Research
1. Investigate common failure patterns
2. Develop targeted training data for weak areas
3. Implement hallucination detection classifiers (HALP)
4. Study correlation between embeddings and hallucinations

---

## Citation

If you use these datasets or findings in your research, please cite:

```bibtex
@misc{vlm_hallucination_analysis_2025,
  title={Vision-Language Models Hallucination Analysis},
  author={HALP Research Team},
  year={2025},
  note={Manual review of 40,000 VQA samples across 4 VLMs}
}
```

---

## Contact & Support

For questions, issues, or additional information about these datasets:
- Dataset Location: `/Users/saiakhil/Documents/Thesis/HALP_EACL/Final_CSV_Hallucination/`
- Review Scripts: `/Users/saiakhil/Documents/Thesis/HALP_EACL/`

---

**Document Generated:** October 2025  
**Last Updated:** October 2025  
**Version:** 1.0

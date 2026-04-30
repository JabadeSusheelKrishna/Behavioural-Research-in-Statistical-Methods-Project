# Research Report: Sentence Memorability and Syntactic Processing

## 1. Hypotheses

### Hypothesis 1: Syntactic Normalization Penalty
*   **Null ($H_0$):** There is no difference in word-recognition (WR) accuracy between sentences repeated in the same voice and sentences where the voice was transformed (Active $\leftrightarrow$ Passive).
*   **Alternate ($H_a$):** WR accuracy will be significantly lower for Transformed sentences, indicating that syntactic surface details are lost while semantic gist is retained.

### H1 Extension: Canonical Form Bias (Voice Asymmetry)
*   **Null ($H_0$):** The accuracy penalty for transforming a sentence is equal regardless of whether the original was Active or Passive.
*   **Alternate ($H_a$):** The penalty will be significantly larger for Active-voice originals, as memory "normalizes" passive inputs toward the active default during storage.

### Hypothesis 2: Memory Strength Model
*   **Null ($H_0$):** Item-recognition reaction time (IR RT) does not predict the accuracy of subsequent wording retrieval.
*   **Alternate ($H_a$):** Faster reaction times will predict higher accuracy, suggesting that a single strong memory signal drives both retrieval speed and detail precision.

---

## 2. Descriptive Observations

*   **Core Normalization Effect:** A clear performance gap was observed. Participants achieved **80.95%** accuracy on Same-voice repeats, which dropped to **68.15%** for Transformed-voice repeats.
*   **Voice Asymmetry:** The "Normalization Penalty" was highly skewed. Flipping an **Active** sentence to Passive caused an **18.4%** drop in accuracy, whereas flipping a **Passive** sentence to Active caused only a **6.5%** drop.
*   **Latency Trends:** Successful word recognition was associated with faster initial detection. The median RT for correct trials was **1578.5ms**, nearly **50ms faster** than the 1627.0ms observed for incorrect trials. Accuracy rates declined steadily from the fastest quartile (77%) to the slowest (71%).

---

## 3. Inferential Statistics & Test Rationales

### Test 1: Paired Wilcoxon Signed-Rank (for H1 and Extension)
*   **Why we chose this:** 
    1.  **Ceiling Effect:** Participant medians were stuck at 1.0, making standard median-based tests insensitive. We used per-participant mean proportions to capture the variance.
    2.  **Non-Normality:** Shapiro-Wilk tests on the differences were significant ($p < .001$), violating the assumptions for a paired t-test.
    3.  **Within-Subjects:** The test accounts for the fact that the same participants provided data for both conditions.
*   **Inferential Result:** 
    *   **H1:** $W = 4521.0, p < .000001$. Large effect size ($r = 0.54$).
    *   **Extension:** $W = 1924.0, p = .0012$. Significant asymmetry confirmed.

### Test 2: Generalized Estimating Equations (GEE) (for H2)
*   **Why we chose this:** 
    1.  **Clustered Data:** Our dataset contains 4,381 trials nested within 112 participants. Standard Logistic Regression assumes every trial is independent, which leads to "pseudoreplication" and false significance.
    2.  **Robust Errors:** GEE with an "Exchangeable" correlation structure correctly adjusts the standard errors to account for the fact that trials from the same person are related.
    3.  **Covariate Control:** It allowed us to test the effect of speed while simultaneously controlling for the difficulty of the sentence condition (Same vs. Transformed).
*   **Inferential Result:** 
    *   $RT\_z$ coefficient $= -0.0818, p = 0.024$. Odds Ratio $= 0.921$.

---

## 4. Final Results

1.  **Syntactic Normalization is Robust:** We **Reject $H_0$** for Hypothesis 1. There is a massive, statistically significant penalty for changing sentence voice in memory. This confirms that human memory is "gist-based"—we remember the meaning but discard the exact syntax.
2.  **Memory Defaults to Active Voice:** We **Reject $H_0$** for the H1 Extension. The significantly larger penalty for Active-voice originals proves the **Canonical Form Bias**. It is harder to detect a change when a sentence moves *away* from the active voice than when it moves *toward* it.
3.  **Faster is Sharper:** We **Reject $H_0$** for Hypothesis 2. Even after correcting for participant clustering, faster recognition significantly predicts higher accuracy. This supports the **Memory Strength** account: a high-quality memory trace is retrieved both rapidly and accurately, whereas a weak trace leads to both hesitation and error.
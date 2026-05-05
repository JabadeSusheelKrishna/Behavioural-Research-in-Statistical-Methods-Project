# 🧠 Sentence Memorability: Behavioral Research in Statistical Methods

Welcome to the **Sentence Memorability Project**! This repository contains the experimental analysis, data processing pipelines, and statistical modeling for investigating how human memory for sentences is influenced by the memorability of constituent words and syntactic structures (Voice).

This project was developed as part of the **BRSM (Behavioral Research in Statistical Methods)** course.

---

## 🌟 Project Overview

The core objective of this study is to quantify how the "memorability" of individual nouns and the syntactic framing of a sentence (Active vs. Passive voice) affect overall sentence recall.

### 🧪 Experimental Design
We employed a **continuous recognition experiment** using simple Subject–Verb–Object (S–V–O) sentences. 

1.  **Word-Level Memorability**: Subject and object nouns were systematically varied between High (H) and Low (L) memorability, resulting in four types:
    *   **HH** (High–High)
    *   **HL** (High–Low)
    *   **LH** (Low–High)
    *   **LL** (Low–Low)
2.  **Syntactic Voice**: Each sentence was presented in either **Active** or **Passive** voice.
3.  **Total Conditions**: 8 conditions (4 Word Types × 2 Voices).
4.  **Participants**: 114 participants (112 after strict attention-check validation).

---

## 🚀 Getting Started

### 📋 Prerequisites
Ensure you have Python 3.8+ installed. You will need the following libraries:
```bash
pip install pandas numpy matplotlib seaborn scipy
```

### 📂 Project Structure
*   `Sentence Memorability/`: Contains raw anonymized participant logs (`.log` files).
*   `Hypothesis Analysis/*`: Has analysis for all the Hypothesis.
*   `results/`: Generated visualizations and CSV summaries (Cleaned/Final) for Mid.
*   `papers/`: Foundational research papers regarding word concreteness and imagery.

---

## 🛠️ Running the Analysis

- You can just the respective notebook files
- go to Hypothesis Directory
```bash
cd "Hypothesis Analysis"
```
- run each of the notebook file or open in colab and click run all

## 📊 Key Findings & Visualizations

This study investigated sentence memorability across 112 participants using a continuous recognition paradigm. Key findings are as follows:
Word memorability type significantly influenced recognition, with LL sentences consistently scoring lower than all other conditions, confirming that even a single high-memorability noun is sufficient to sustain recognition performance. Grammatical voice, however, did not affect recognition accuracy but did impose a measurable retrieval cost, with passive sentences taking significantly longer to recognise. WR accuracy showed a descriptively consistent but statistically non-significant trend across memorability conditions, likely reflecting a power limitation.
Contrary to the classic speed-accuracy tradeoff, faster responses were significantly more accurate, pointing to a decisiveness effect driven by memory strength rather than a speed-accuracy exchange. Performance improved rather than declined across experimental blocks, indicating a learning effect. Syntactic transformation of voice produced a substantial accuracy penalty in wording retrieval, confirming that surface-level syntactic form is retained in memory. Finally, within-subject RT fluctuations predicted wording retrieval accuracy in the unadjusted model, though this effect was partially explained by sentence-level difficulty.
Overall, sentence memorability is primarily shaped by noun-level memorability, while grammatical voice influences retrieval efficiency rather than recognition accuracy. These results support Dual Coding Theory and psycholinguistic accounts of passive processing, while also revealing a memory-strength-driven decisiveness effect in recognition behaviour.


---

## 👤 Authors

- Susheel Krishna
- Pavan Karke
- Vishnu Varun


---

## 📜 License
This project is for academic purposes as part of the BRSM course curriculum.

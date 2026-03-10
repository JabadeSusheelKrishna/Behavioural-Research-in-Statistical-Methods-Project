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
*   `sentence_memorability_report_3.py`: The primary analysis script (Latest version).
*   `results2/`: Generated visualizations and CSV summaries (Cleaned/Final).
*   `results/`: Initial/Raw plot outputs.
*   `papers/`: Foundational research papers regarding word concreteness and imagery.

---

## 🛠️ Running the Analysis

The analysis follows a two-step workflow: generating the processed metrics and then creating refined visualizations.

### Step 1: Process Logs & Generate Metrics
This command parses the raw `.log` files, performs block validation, and calculates corrected memorability scores.

```bash
python sentence_memorability_report_3.py --logs_dir "./Sentence Memorability/NewLogsAnonymized" --output_dir "./results2"
```

**What this step does:**
*   **Data Cleaning**: Removes practice blocks and filters irrelevant events.
*   **Validation**: Implements the exclusion formula: `Correct IRs > (Wrong IRs / 2) + Missed IRs`. 
*   **Metrics Calculation**: Computes **Corrected Memorability Scores (CMS)**, Reaction Times (RT), and Word Recognition (WR) Accuracy.
*   **Outputs**: Saves `corrected_memorability.csv` which is required for the next step.

### Step 2: Generate Refined Visualizations
Once the CSV is generated, run the updated plotting script to create high-quality, publication-ready figures.

```bash
python report2_updated_plots.py
```

**What this step does:**
*   Generates cleaned bar charts with 95% Confidence Intervals.
*   Produces interaction plots and violin plots for distribution analysis.
*   Saves outputs to the `results2/` folder.

---

## 📊 Key Findings & Visualizations

The project generates comprehensive visualizations located in the `results2/` folder:

*   **Raincloud Plots**: Show the raw distribution, boxplot, and density of CMS across word types and voices.
*   **Interaction Plots**: Visualize how Voice affects different word-memorability types differently.
*   **Normality Checks**: Q-Q plots and Histogram panels showing that the data is statistically non-normal ($p < 0.001$), necessitating non-parametric tests.
*   **Reaction Latency**: Analysis of how abstract syntax (passive voice) or abstract words (LL) slow down recognition time.

---

## 👤 Author

**Jabade Susheel Krishna**  
Computer Science Undergraduate, IIIT Hyderabad  
[GitHub](https://github.com/JabadeSusheelKrishna) | [Email](mailto:susheelkrishnajabade@gmail.com)

---

## 📜 License
This project is for academic purposes as part of the BRSM course curriculum.

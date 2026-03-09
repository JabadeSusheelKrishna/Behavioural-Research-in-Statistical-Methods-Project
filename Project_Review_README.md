# 🔬 Sentence Memorability Project: Detailed Review & Feedback

Hi Team! I have thoroughly reviewed the experimental details, the raw logs, the Python scripts in your `Null Hypothesis Testing` folder, and the initial report (`BRSM_Project_Report - 1.pdf`). 

Assuming that testing for non-normality (Shapiro-Wilk) and using non-parametric tests (Kruskal-Wallis, Mann-Whitney) are out of scope for now and reserved for future updates, your approach of using **paired t-tests** perfectly matches your within-subject design experiments! 

Your `null_hyp1.py` and `null_hyp2.py` scripts effectively run these tests, and the outputs perfectly match the statistical numbers presented in the PDF. However, there are still **critical methodological bugs** in the hypothesis test code compared to your report text, as well as untouched data that can make your project stand out.

Here is a detailed breakdown of issues, missing analyses, and actionable recommendations.

---

## 🚨 1. Critical Discrepancies Between Code and Report

### A. Missing Block Validation in Hypothesis Scripts
* **The Issue:** In your PDF report (Section 2.4), you rigorously state that blocks were excluded based on a strict validation criterion: `Correct > (Wrong / 2) + Missed`. You clearly note that two participants were fully excluded due to this, leaving **112** participants in the final dataset for analysis.
* **The Error:** Your files in the `Null Hypothesis Testing` folder (`null_hyp1.py` and `null_hyp2.py`) completely lack this exclusion logic! If you look at the console output commented at the bottom of those scripts, it says exactly `Total valid records: 456` and `Processing complete! Records found: 228`. This corresponds to **114** participants (114 * 4 conditions = 456 records).
* **The Fix:** You need to port the block segmenting and validation logic into `null_hyp1.py` and `null_hyp2.py`. Right now, your paired t-test results in the report are tainted by data from inattentive participants, and the script's output contradicts the report's text!

### B. Practice Block Leakage
* **The Issue:** While you use `~df['Event'].astype(str).str.contains('Practice|gap_time', case=False, na=False)` to remove practice rows, you need to ensure that the initial non-practice blocks are properly synced with the `Rest Phase started` markers. Currently, without full block parsing, you are globbing all rows together, risking inaccurate False Alarm rate calculations.

---

## 💡 2. Missing Variables & Analysis Gaps

By parsing your raw log files (e.g., `232.log`), I found extremely valuable data columns that you are entirely ignoring. These represent major missed opportunities to enhance your report.

### A. Reaction Time (The biggest missing piece)
* **The Missing Variable:** The logs contain `Reaction_time_IR` (Initial Recognition Reaction Time) and `Reaction_time_WR` (Word Recognition Reaction Time).
* **Why it matters:** In behavioral research, examining memory latency (how fast someone retrieves a memory) is just as important as accuracy. 
* **Actionable Idea:** Do participants take significantly longer to correctly recognize an LL (abstract) sentence compared to an HH (concrete) sentence? Does processing a Passive sentence slow down the reaction time compared to an Active sentence?

### B. Correct/Incorrect Word Recognition (Second Task)
* **The Missing Variable:** The dataset has an `Accuracy WR` (Word Recognition) column and associated button presses ("Yes" / "No") mapped to `WR pressed` events.
* **Why it matters:** You only analyzed the Initial Recognition (IR) hit rate. What about the secondary Word Recognition task? Is it harder to correctly answer the secondary WR prompt for a low-memorability sentence?

---

## 📈 3. Expanding Hypotheses

You correctly noted that you have "less hypothesis statements". Here is how you can expand them to make the project much richer while easily sticking to your paired t-tests:

*   **Hypothesis 3 (Position Effect):** *Does the position of the concrete/high-memorability word matter?* Compare `HL` vs `LH` with a paired t-test. Does having a highly memorable Subject (`HL`) make a sentence more memorable than a highly memorable Object (`LH`)? (Your descriptive stats show a subtle difference: HL=0.730 vs LH=0.718).
*   **Hypothesis 4 (Latency / Cognitive Load):** *Abstract concepts take longer to mentally retrieve.* Null Hypothesis: Reaction time for correct hits in LL sentences equals HH sentences. Alternative: Reaction time for LL sentences is significantly higher (slower) than for HH sentences.
*   **Hypothesis 5 (Interaction Check):** Check if Voice affects different word types differently using paired t-tests (e.g. comparing the difference in `Passive - Active` for HH against the difference in LL).

---

## 📊 4. Fixing Visualizations (Less Cognitive-Heavy)

Your report currently uses Box Plots for standard visualization, and your scripts overlay random 25%-30% data points. Providing clear, non-overwhelming visuals is crucial.

**Recommendations for lighter visual load:**
1.  **Simple Bar Charts with Error Bars:** While overlapping points (Saif’s suggestion) provides a raw look at data, a clean Bar Chart showing Means and Error Bars (representing Standard Error of the Mean, or 95% Confidence Intervals) is the most universally understood format in psychology research. It removes cognitive load entirely.
2.  **Interaction Plot Keep It Simple:** The interaction line plot is much easier to read than box plots for multiple variables. Ensure the lines have clear markers and distinct colors (e.g., Blue for Active, Red for Passive).

---

### ✅ Summary Checklist for Next Steps:
- [ ] Migrate the **Block Validation** logic into `null_hyp1.py` and `null_hyp2.py` so the tests only run on the 112 valid participants.
- [ ] Recalculate your paired t-tests after excluding the invalid blocks and update the numbers in your final report.
- [ ] Extract `Reaction_time_IR` from the logs and test Hypothesis 4 (Latency).
- [ ] Perform a paired t-test comparing `HL` vs `LH`.
- [ ] Clean up charts by shifting away from random point overlays if they look too chaotic, or moving toward clean Confidence Interval Bar charts.

# 📝 Modification Report for `BRSM_Project_Report_mid_text_format.txt`

After meticulously reviewing the current draft, the background materials (`Sentence Memorability.txt` and the `papers/` folder), and our updated scripts/plots, here is the exact list of modifications you need to make to your report.

---

### 1. Introduction Section
*   **Fix the Citations:** The report currently cites `(Rubin & Friendly, 1986; Shepard, 1967)`. However, the papers in your `papers/` folder correspond to completely different years (1968, 1971, 1973—likely Paivio's work on concreteness and imagery). You need to update the citations in the text to match the actual PDFs provided in the `papers/` folder.
*   **Improve Phrasing:** Change *"According to the research, the paper wants to experiment how humans memorize sentences"* to something more professional, such as: *"This study investigates how sentence memorability operates, specifically examining how it is influenced by the memorability of constituent words and syntactic framing."*

### 2. Method Overview Section
*   **Alignment Check: ✅ VERIFIED.** The participant count (114 total, passing 112) is correct. The independent/dependent variables are correct. The validation logic `Correct validation IRs > (Wrong Validation IRs ÷ 2) + Missed Validation IRs` accurately reflects what we coded.
*   **Modification:** No major content changes needed here. The methodology perfectly aligns with `sentence_memorability_report1.py`.

### 3. Data Preparation Section
*   **Alignment Check: ✅ VERIFIED.** The block mapping, boolean flag conversions, and attention checks (7 blocks excluded, leaving 112 participants) exactly match our Python script outputs.
*   **Modification:** The text is accurate and robust. No changes needed.

### 4. Descriptive Statistics & Results (CRITICAL UPDATES)
You have generated completely new, cleaned plots in `results2`. The current text refers to "boxplots" and has redundant information. 

*   **Section 4.1 (Word Type):**
    *   **Change Text:** You currently wrote, *"Figure 1 shows boxplots for each condition."* Change this to: *"Figure 1 shows a bar chart of the mean corrected memorability for each condition, with error bars representing the 95% Confidence Intervals."*
    *   **Replace Image:** Add **`results2/fig1_cleaned_wordtype.png`** here.
*   **Section 4.2 (Memorability by Voice):**
    *   **Change Text:** Update references to the visual distributions to match the new bar chart format.
    *   **Replace Image:** Add **`results2/fig2_cleaned_voice.png`** here.
*   **Section 4.3 (Interaction Exploration):**
    *   **Replace Image:** Add **`results2/fig3_cleaned_interaction.png`** here.
*   **NEW SECTION 4.4: Normality Assumptions (Shapiro-Wilk Test)**
    *   **Add Text:** You must insert a new paragraph stating: *"Before proceeding with inferential statistics, we tested the assumption of normality using the Shapiro-Wilk test. The corrected memorability scores exhibited a heavily skewed distribution, and the Shapiro-Wilk tests for all conditions were statistically significant (p < 0.001), indicating that the data is strictly non-normal."*
    *   **Add Images:** Insert **`results2/fig6_distribution_hist.png`** (to show the overall skew) and **`results2/fig4_shapiro_panel.png`** (to show the breakdown by condition) in this new subsection.

### 5. Inferential Statistics & Conclusion
*   **CRITICAL DELETION:** **Remove Section 5 entirely!**
    *   **Reasoning:** As you correctly pointed out, the paired t-tests assume a normal distribution. Because our Shapiro-Wilk test proved the data is non-normal, applying a paired t-test is a methodological error.
*   **Update the Conclusion/Future Work:**
    *   Remove the claims in the conclusion about definitively rejecting/failing to reject the null hypotheses using t-tests.
    *   **Add Future Work Text:** *"Because the data violates the assumption of normality, parametric tests (like the paired t-test) are inappropriate. For future work, we will conduct formal hypothesis testing using non-parametric alternatives, such as the Wilcoxon Signed-Rank test or Friedman's ANOVA, to robustly evaluate the statistical significance of the differences observed in our descriptive analysis."*

---

### Summary Checklist for You:
- [ ] Fix introductory citations to match the 1968, 1971, 1973 papers.
- [ ] Swap Figure 1, 2, and 3 with the ones named `_cleaned_` from the `results2` folder.
- [ ] Create Section 4.4 and insert the exact Shapiro-Wilk results and distribution plots (`fig4` and `fig6`).
- [ ] **DELETE Section 5 (T-tests).**
- [ ] Add the non-parametric goal to the Future Work section in the conclusion.

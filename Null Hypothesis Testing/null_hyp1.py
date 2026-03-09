import os
import glob
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. DIRECTORY SETUP & DATA PROCESSING 
# ---------------------------------------------------------
log_dir = r"E:\windows_down\Mid\Mid\Sentence Memorability\NewLogsAnonymized"
log_files = glob.glob(os.path.join(log_dir, "*.log"))

print(f"Found {len(log_files)} log files. Processing...")

# Mapping based on the log file formatting!
STIM_MAP = {"HH": "HH", "HVL": "HL", "LVH": "LH", "LVL": "LL"}

records =[]

for file in log_files:
    participant_id = os.path.basename(file).split('.')[0]
    
    try:
        df = pd.read_csv(file, low_memory=False)
    except Exception as e:
        print(f"Could not read {file}: {e}")
        continue
        
    # Remove Practice rows and gap_time
    df = df[~df['Event'].astype(str).str.contains('Practice|gap_time', case=False, na=False)].reset_index(drop=True)
    
    # Convert flag columns to boolean
    for col in ['isTarget', 'isValidation', 'isRepeat']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower() == 'true'
            
    fa_count = 0
    non_repeat_count = 0
    hits = {'HH': 0, 'HL': 0, 'LH': 0, 'LL': 0}
    misses = {'HH': 0, 'HL': 0, 'LH': 0, 'LL': 0}
    
    for i, row in df.iterrows():
        if row['Event'] == 'Sentence shown':
            stimulus = str(row['Stimulus'])
            if '_' not in stimulus: 
                continue
            
            # Extract word type using the STIM_MAP to fix the HVL/LVH/LVL mismatch
            prefix = stimulus.split('_')[0] 
            word_type = STIM_MAP.get(prefix, None)
            
            # Skip if it's a filler or practice sentence not in our main 4 types
            if word_type is None:
                continue
            
            # Check for Target sentences that are NOT validations
            if row.get('isTarget', False) and not row.get('isValidation', False):
                
                # Check up to 3 subsequent rows for 'IR pressed'
                ir_pressed = False
                for j in range(1, 4):
                    if i + j < len(df):
                        next_event = str(df.loc[i+j, 'Event'])
                        if 'IR pressed' in next_event:
                            ir_pressed = True
                            break
                        if 'Sentence shown' in next_event:
                            break
                
                # Update Hits, Misses
                if row.get('isRepeat', False):
                    if ir_pressed:
                        hits[word_type] += 1
                    else:
                        misses[word_type] += 1
                else:
                    # Update False Alarms
                    non_repeat_count += 1
                    if ir_pressed:
                        fa_count += 1
                        
    # Calculate Global False Alarm (FA) Rate for this participant
    fa_rate = fa_count / non_repeat_count if non_repeat_count > 0 else 0
    
    # Calculate Corrected Memorability (Hit Rate - FA Rate) per condition
    for wt in ['HH', 'HL', 'LH', 'LL']:
        total_targets = hits[wt] + misses[wt]
        if total_targets > 0:
            hit_rate = hits[wt] / total_targets
            corr_mem = hit_rate - fa_rate
            records.append({
                'Participant_ID': participant_id,
                'Word_Type': wt,
                'Corrected_Memorability': corr_mem
            })

# Create the final processed DataFrame
final_df = pd.DataFrame(records)
print(f"Data processing complete! Total valid records: {len(final_df)}\n")

# ---------------------------------------------------------
# 2. HYPOTHESIS TESTING (Statement-1: Word Memorability)
# ---------------------------------------------------------
# H0: LL >= HH, HL, LH
# Ha: LL < HH, HL, LH

print("=== Hypothesis 1 Test Results (LL vs others) ===")

# Pivot data to ensure paired testing works correctly per participant
pivot_df = final_df.pivot(index='Participant_ID', columns='Word_Type', values='Corrected_Memorability').dropna()

ll_scores = pivot_df['LL'].values
hh_scores = pivot_df['HH'].values
hl_scores = pivot_df['HL'].values
lh_scores = pivot_df['LH'].values

# Perform Paired t-tests (Alternative='less' checks if LL is significantly LOWER than others)
t_hh, p_hh = stats.ttest_rel(ll_scores, hh_scores, alternative='less')
t_hl, p_hl = stats.ttest_rel(ll_scores, hl_scores, alternative='less')
t_lh, p_lh = stats.ttest_rel(ll_scores, lh_scores, alternative='less')

print(f"LL vs HH: t-statistic = {t_hh:.3f}, p-value = {p_hh:.5e}")
print(f"LL vs HL: t-statistic = {t_hl:.3f}, p-value = {p_hl:.5e}")
print(f"LL vs LH: t-statistic = {t_lh:.3f}, p-value = {p_lh:.5e}")
print("================================================\n")

# ---------------------------------------------------------
# 3. PLOT GRAPH (Saif's suggestion: Overlapping Random Points)
# ---------------------------------------------------------
plt.figure(figsize=(9, 6))

# Base Boxplot using the exact colors from your report theme
PAL = {"HH": "#2E86AB", "HL": "#F18F01", "LH": "#4CAF50", "LL": "#C73E1D"}
sns.boxplot(x='Word_Type', y='Corrected_Memorability', data=final_df, 
            order=['HH', 'HL', 'LH', 'LL'], palette=PAL, showfliers=False, width=0.5)

# Sample random points (e.g., 25% of the data) for the overlay (Saif's request)
np.random.seed(42) # Ensure the graph looks the same every time it is run
df_sampled = final_df.sample(frac=0.25)

# Overlay the randomly chosen points using stripplot
sns.stripplot(x='Word_Type', y='Corrected_Memorability', data=df_sampled, 
              order=['HH', 'HL', 'LH', 'LL'], color='black', alpha=0.5, jitter=True, size=4)

# Formatting
plt.title("Corrected Memorability by Word Type\n(with randomly sampled participant scores overlaid)", fontsize=12, fontweight='bold')
plt.ylabel("Corrected Memorability Score (Hit Rate - FA Rate)", fontsize=11)
plt.xlabel("Word Memorability Condition", fontsize=11)
plt.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Set custom x-axis labels to match the report style
plt.xticks(ticks=[0, 1, 2, 3], labels=["HH\n(High-High)", "HL\n(High-Low)", "LH\n(Low-High)", "LL\n(Low-Low)"])

plt.tight_layout()
plt.show()




"""
Found 114 log files. Processing...
Data processing complete! Total valid records: 456

=== Hypothesis 1 Test Results (LL vs others) ===
LL vs HH: t-statistic = -3.781, p-value = 1.25432e-04
LL vs HL: t-statistic = -4.629, p-value = 4.92756e-06
LL vs LH: t-statistic = -2.878, p-value = 2.39123e-03
================================================

"""
"""

This is a perfect result! The script processed all 114 participants correctly (
114
 participants
×
4
 word types
=
456
 valid records
114 participants×4 word types=456 valid records
), and the statistics output tells a very clear story.
Here is a simple, complete explanation of the hypothesis, the results, and exactly what you should write in your report based on this code's output.
1. Understanding the Output Numbers
The code performed three Paired one-tailed t-tests. It compared the scores of LL (Low subject, Low object) against the other three categories (HH, HL, LH) to see if LL is statistically significantly lower.
Scientific Notation (e-0X): The e-0X at the end of your p-values means you move the decimal point to the left 
X
X
 times.
1.25432e-04 = 
0.000125
0.000125
4.92756e-06 = 
0.00000493
0.00000493
2.39123e-03 = 
0.00239
0.00239
The Rule: In psychology and behavioral research, if a p-value is less than 0.05, the result is considered "Statistically Significant."
The t-statistic: The negative numbers (e.g., 
−
3.781
−3.781
) prove the direction. It mathematically confirms that the LL average was lower than the HH, HL, and LH averages.
2. The Hypothesis Conclusion
The Null Hypothesis (
H
0
H 
0
​
 
): Asserted that LL sentences are not significantly lower than HH, HL, or LH.
The Alternative Hypothesis (
H
a
H 
a
​
 
): Asserted that LL sentences are significantly lower.
Result: Because all three p-values are way below 0.05, we have overwhelming evidence to Reject the Null Hypothesis (
H
0
H 
0
​
 
) and Accept the Alternative Hypothesis (
H
a
H 
a
​
 
). Sentences with Low-Memorability subjects and objects are significantly harder for humans to remember.
3. What to Paste into Your Report (Page 6)
You can directly copy and paste these sections into the bottom of Page 6 of your project report where you have the yellow highlighted placeholders.
— Inferential Statistics —
To formally test Statement-1, a series of one-tailed paired t-tests were conducted to compare the Corrected Memorability scores of the Low-Low (LL) condition against the HH, HL, and LH conditions. The tests confirmed that memorability scores for LL sentences were significantly lower than HH sentences (
t
=
−
3.78
,
p
<
0.001
t=−3.78,p<0.001
), HL sentences (
t
=
−
4.63
,
p
<
0.001
t=−4.63,p<0.001
), and LH sentences (
t
=
−
2.88
,
p
=
0.002
t=−2.88,p=0.002
).
— Conclusion —
Based on the descriptive data and the inferential paired t-tests, we reject the Null Hypothesis (
H
0
H 
0
​
 
) for Statement-1. Sentences constructed with abstract, semantically impoverished nouns (LL condition) yield significantly lower memorability scores compared to sentences containing at least one high-memorability noun (HH, HL, LH). Furthermore, as seen in our preliminary analysis of Voice, active vs. passive framing has minimal impact compared to the memorability of the constituent words.
"""
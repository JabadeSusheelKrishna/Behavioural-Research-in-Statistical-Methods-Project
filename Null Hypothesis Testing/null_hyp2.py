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

print(f"Found {len(log_files)} log files. Processing Voice Conditions...")

records_voice = []

for file in log_files:
    participant_id = os.path.basename(file).split('.')[0]
    
    try:
        df = pd.read_csv(file, low_memory=False)
    except Exception as e:
        continue
        
    # Remove Practice rows and gap_time
    df = df[~df['Event'].astype(str).str.contains('Practice|gap_time', case=False, na=False)].reset_index(drop=True)
    
    # Convert flag columns to boolean
    for col in ['isTarget', 'isValidation', 'isRepeat']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower() == 'true'
            
    fa_count = 0
    non_repeat_count = 0
    hits = {'Active': 0, 'Passive': 0}
    misses = {'Active': 0, 'Passive': 0}
    
    for i, row in df.iterrows():
        if row['Event'] == 'Sentence shown':
            stimulus = str(row['Stimulus'])
            if '_' not in stimulus: 
                continue
            
            # Determine Voice: _A is Active, _P is Passive
            if stimulus.endswith('_A'):
                voice = 'Active'
            elif stimulus.endswith('_P'):
                voice = 'Passive'
            else:
                continue
            
            # Target sentences (exclude validations)
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
                
                if row.get('isRepeat', False):
                    if ir_pressed:
                        hits[voice] += 1
                    else:
                        misses[voice] += 1
                else:
                    non_repeat_count += 1
                    if ir_pressed:
                        fa_count += 1
                        
    fa_rate = fa_count / non_repeat_count if non_repeat_count > 0 else 0
    
    for v in ['Active', 'Passive']:
        total = hits[v] + misses[v]
        if total > 0:
            hit_rate = hits[v] / total
            records_voice.append({
                'Participant_ID': participant_id,
                'Voice': v,
                'Corrected_Memorability': hit_rate - fa_rate
            })

voice_df = pd.DataFrame(records_voice)
print(f"Processing complete! Records found: {len(voice_df)}\n")

# ---------------------------------------------------------
# 2. HYPOTHESIS TESTING (Statement-2: Sentence Voice)
# ---------------------------------------------------------
# H0: Passive == Active (No difference)
# Ha: Passive < Active (Passive is lower)

print("=== Hypothesis 2 Test Results (Active vs Passive) ===")

pivot_voice = voice_df.pivot(index='Participant_ID', columns='Voice', values='Corrected_Memorability').dropna()

active_scores = pivot_voice['Active'].values
passive_scores = pivot_voice['Passive'].values

# Mean calculation for context
print(f"Mean Memorability (Active): {np.mean(active_scores):.4f}")
print(f"Mean Memorability (Passive): {np.mean(passive_scores):.4f}")

# Perform Paired t-test (one-tailed, alternative 'less' checks if Passive < Active)
t_stat, p_val = stats.ttest_rel(passive_scores, active_scores, alternative='less')

print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_val:.4f}")
print("====================================================\n")

# ---------------------------------------------------------
# 3. PLOT GRAPH (Random Points Overlaid)
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))

# Boxplot
sns.boxplot(x='Voice', y='Corrected_Memorability', data=voice_df, 
            palette=['#3498db', '#95a5a6'], showfliers=False, width=0.4)

# Sample 30% of random points to overlay
np.random.seed(42)
voice_sampled = voice_df.sample(frac=0.30)

# Overlay points
sns.stripplot(x='Voice', y='Corrected_Memorability', data=voice_sampled, 
              color='black', alpha=0.4, jitter=True, size=5)

plt.title("Effect of Sentence Voice on Memorability\n(Randomly Sampled Scores Overlaid)", fontsize=12, fontweight='bold')
plt.ylabel("Corrected Memorability Score")
plt.xlabel("Grammatical Structure")
plt.axhline(0, color='red', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()



"""
E:\windows_down\Mid\Mid> python .\null_hyp2.py
Found 114 log files. Processing Voice Conditions...
Processing complete! Records found: 228

=== Hypothesis 2 Test Results (Active vs Passive) ===
Mean Memorability (Active): 0.7780
Mean Memorability (Passive): 0.7904
T-statistic: 1.330
P-value: 0.9069
"""

"""

"""
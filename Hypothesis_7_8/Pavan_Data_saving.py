import os, csv, glob, re
import pandas as pd
import numpy as np

# ── CONFIG ─────────────────────────────────────────────────────────────────────
LOG_DIR      = r'NewLogsAnonymized'
OUT_H1       = 'H1_Normalization'
OUT_H2       = 'H2_Familiarity'

RT_MIN, RT_MAX = 200, 4000   # standard cognitive science outlier bounds

COND_MAP = {'HH': 'HH', 'HVL': 'HL', 'LVH': 'LH', 'LVL': 'LL'}

os.makedirs(OUT_H1, exist_ok=True)
os.makedirs(OUT_H2, exist_ok=True)


def validate_block(blk):
    """
    Returns True if the block passes the attentiveness criterion:
        Correct IR > (Wrong IR / 2) + Missed IR
    """
    # 1. Count the explicit events
    correct = sum(1 for r in blk if r.get('Event') == 'Validation IR pressed')
    wrong   = sum(1 for r in blk if r.get('Event') == 'Validation Wrong IR pressed')
    missed  = sum(1 for r in blk if r.get('Event') == 'Validation Missed')
    
    # 2. Apply the formula
    # We use (wrong / 2) to penalize 'button mashing' but not as harshly as a total miss
    return correct > (wrong / 2) + missed


def get_word_type(stimulus):
    m = re.match(r'^([A-Z]+)', str(stimulus))
    if not m: return None
    return COND_MAP.get(m.group(1))


def get_voice(stimulus):
    s = str(stimulus)
    if s.endswith('_A'): return 'Active'
    if s.endswith('_P') or s.endswith('P'): return 'Passive'
    return None


def derive_condition(btn, acc_wr):
    """
    Derives the TRUE experimental condition from response + accuracy.
    
    Logic:
        Button=Yes, Acc=1  →  Same       (correct: said same,  was same)
        Button=No,  Acc=0  →  Same       (wrong:   said diff,  was same)
        Button=No,  Acc=1  →  Transformed (correct: said diff, was transformed)
        Button=Yes, Acc=0  →  Transformed (wrong:   said same, was transformed)

    Using Button alone (as in the previous chat code) is WRONG because
    it conflates the participant's response with the ground-truth condition.
    """
    try:
        acc = int(float(acc_wr))
    except (ValueError, TypeError):
        return None
    if btn not in ('Yes', 'No'):
        return None
    if (btn == 'Yes' and acc == 1) or (btn == 'No' and acc == 0):
        return 'Same'
    else:
        return 'Transformed'


# ── MAIN PARSER ─────────────────────────────────────────────────────────────────

def process_file(filepath):
    # 1. Read file
    try:
        with open(filepath, encoding='utf-8-sig') as f:
            raw = list(csv.DictReader(f))
    except UnicodeDecodeError:
        with open(filepath, encoding='latin-1') as f:
            raw = list(csv.DictReader(f))

    if not raw:
        return None

    # Clean whitespace from headers and values
    rows = [{k.strip(): (v.strip() if v else '') for k, v in r.items()} for r in raw]

    # Get participant ID
    try:
        pid = int(float(rows[0].get('participant_ID', -1)))
    except (ValueError, TypeError):
        return None


    # 2. Remove practice and gap_time rows (keep main experiment only)
    rows = [r for r in rows
            if 'Practice' not in r.get('Event', '')
            and 'gap_time'  not in r.get('Event', '')]

    # 3. Split into blocks at every "Rest Phase started" event
    blocks, current_block = [], []
    for r in rows:
        if r.get('Event') == 'Rest Phase started':
            if current_block:
                blocks.append(current_block)
            current_block = []
        else:
            current_block.append(r)
    if current_block:
        blocks.append(current_block)

    # 4. Keep only rows from blocks that PASS the validation criterion
    valid_rows = []
    n_valid, n_invalid = 0, 0
    for blk in blocks:
        if validate_block(blk):
            valid_rows.extend(blk)
            n_valid += 1
        else:
            n_invalid += 1

    if not valid_rows:
        print(f"  PID {pid}: all {len(blocks)} blocks failed validation — excluded")
        return None

    # 5. Extract trial-level data from valid rows
    #    Trigger on WR pressed; look back for the matching IR pressed
    trials = []
    for i, row in enumerate(valid_rows):

        if row.get('Event') != 'WR pressed':
            continue

        # Must be a main-experiment target repeat
        if row.get('isTarget','').lower() not in ('true','1'):
            continue
        if row.get('isRepeat','').lower() not in ('true','1'):
            continue
        if row.get('isValidation','').lower() in ('true','1'):
            continue

        btn      = row.get('Button')
        acc_wr_r = row.get('Accuracy WR')

        try:
            acc_wr = int(float(acc_wr_r))
        except (ValueError, TypeError):
            continue

        condition = derive_condition(btn, acc_wr)
        if condition is None:
            continue

        # Look back up to 6 rows to find the matching IR pressed event
        stimulus = row.get('Stimulus')
        rt_ir  = None
        acc_ir = None

        for j in range(i - 1, max(i - 7, -1), -1):
            prev = valid_rows[j]
            if prev.get('Event') == 'Sentence shown':
                break   # crossed a trial boundary, stop
            if prev.get('Event') == 'IR pressed' and prev.get('Stimulus') == stimulus:
                try:
                    rt_ir  = float(prev.get('Reaction_time_IR'))
                    acc_ir = int(float(prev.get('Accuracy IR')))
                except (ValueError, TypeError):
                    pass
                break

        # Only include Hits (participant correctly pressed Spacebar on a repeat)
        if acc_ir != 1:
            continue

        try:
            rt_wr = float(row.get('Reaction_time_WR'))
        except (ValueError, TypeError):
            rt_wr = None

        trials.append({
            'PID'        : pid,
            'Stimulus'   : stimulus,
            'word_type'  : get_word_type(stimulus),
            'voice'      : get_voice(stimulus),
            'Condition'  : condition,      # Same or Transformed (ground truth)
            'Accuracy_WR': acc_wr,         # 1 = correct WR response, 0 = incorrect
            'RT_IR'      : rt_ir,          # ms from sentence onset to Spacebar
            'RT_WR'      : rt_wr,          # ms from Spacebar to Yes/No
            'n_valid_blocks'  : n_valid,
            'n_invalid_blocks': n_invalid,
        })

    return trials


# ── RUN OVER ALL LOG FILES ───────────────────────────────────────────────────────

print(f"Scanning: {LOG_DIR}\n")
log_files = sorted(glob.glob(os.path.join(LOG_DIR, '*.log')))
print(f"Found {len(log_files)} log files\n")

all_trials = []
for fp in log_files:
    result = process_file(fp)
    if result:
        all_trials.extend(result)

df = pd.DataFrame(all_trials)
print(f"\nTotal trials extracted : {len(df)}")
print(f"Participants           : {df['PID'].nunique()}")
print(f"Condition counts       : {df['Condition'].value_counts().to_dict()}")
print(f"Missing RT_IR          : {df['RT_IR'].isna().sum()}")
print(f"Missing RT_WR          : {df['RT_WR'].isna().sum()}")

# ── SAVE H1 DATA ────────────────────────────────────────────────────────────────
# H1 needs: Condition (Same/Transformed) and WR accuracy
# No RT filter needed — RT is not the variable of interest for H1

h1_cols = ['PID', 'Stimulus', 'word_type', 'voice', 'Condition', 'Accuracy_WR', 'RT_WR']
df_h1 = df[h1_cols].dropna(subset=['Condition', 'Accuracy_WR']).copy()
df_h1.to_csv(os.path.join(OUT_H1, 'h1_normalization.csv'), index=False)
print(f"\nH1 saved → {OUT_H1}/h1_normalization.csv  ({len(df_h1)} rows)")
print(df_h1['Condition'].value_counts())


# ── SAVE H2 DATA ────────────────────────────────────────────────────────────────
# H2 needs: RT_IR (predictor), Accuracy_WR (outcome), Condition + voice (covariates)
# Apply RT outlier filter here

df_h2 = df[['PID', 'Stimulus', 'word_type', 'voice', 'Condition', 'Accuracy_WR', 'RT_IR']]\
          .dropna(subset=['RT_IR', 'Accuracy_WR', 'Condition'])\
          .copy()
df_h2 = df_h2[(df_h2['RT_IR'] >= RT_MIN) & (df_h2['RT_IR'] <= RT_MAX)]
df_h2.to_csv(os.path.join(OUT_H2, 'h2_familiarity.csv'), index=False)
print(f"\nH2 saved → {OUT_H2}/h2_familiarity.csv   ({len(df_h2)} rows)")
print(f"RT_IR range: {df_h2['RT_IR'].min():.0f}ms – {df_h2['RT_IR'].max():.0f}ms")



"""
To ensure that the analyzed reaction times reflect genuine cognitive processing rather than mechanical or attentional artifacts,
   standard preprocessing steps were applied. 
   A lower-bound cutoff of 200 ms was used, consistent with Robert Whelan, 
   who notes that human reaction times have a physiological minimum (~100–200 ms) required for stimulus perception and
   motor execution; responses below this threshold are typically fast guesses and do not reflect the cognitive process 
   of interest. Additionally, an upper-bound cutoff of 4000 ms was applied to exclude excessively slow responses likely arising
   from inattention or distraction. Finally, analyses were restricted to correctly recognized trials (hits) to ensure that reaction 
   times correspond to successful memory retrieval, as incorrect responses (misses or false alarms) may reflect guessing or 
   failure of recognition rather than the underlying cognitive process being studied.
"""
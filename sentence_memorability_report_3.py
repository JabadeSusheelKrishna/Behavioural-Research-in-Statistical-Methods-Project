"""
BRSM Report 1 — Sentence Memorability
Plot Generation Code — All 7 Figures

Usage:
    python mid_report.py --logs_dir "./Sentence Memorability/NewLogsAnonymized" --output_dir ./results

Requirements:
    pip install pandas numpy matplotlib seaborn scipy

Outputs (saved to output_dir):
    fig1_cms_by_wordtype.png    — Boxplot: CMS by word memorability type
    fig2_cms_by_voice.png       — Boxplot: CMS by sentence voice
    fig3_interaction.png        — Interaction plot: Word Type × Voice
    fig4_violin_cms.png         — Violin: CMS distribution per condition
    fig5_wr_accuracy.png        — Boxplot: WR Accuracy by word type
    fig6_ir_rt.png              — Violin: IR Reaction Time by word type
    fig7_qqplots.png            — Q-Q plots: Normality check (4 panels)
"""

import os
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

# ─── CONFIG ──────────────────────────────────────────────────────────────────
COND_MAP    = {'HH': 'HH', 'HVL': 'HL', 'LVH': 'LH', 'LVL': 'LL'}
COND_ORDER  = ['HH', 'HL', 'LH', 'LL']
COND_LABELS = ['HH\n(High–High)', 'HL\n(High–Low)', 'LH\n(Low–High)', 'LL\n(Low–Low)']
PALETTE     = {'HH': '#2E86AB', 'HL': '#E07A5F', 'LH': '#3BB273', 'LL': '#7B4F99'}
COLORS      = [PALETTE[w] for w in COND_ORDER]
VOICE_PAL   = {'A': '#2E86AB', 'P': '#9B5DE5'}

sns.set_theme(style='whitegrid', font='DejaVu Sans')
plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'figure.dpi': 150, 'savefig.dpi': 150,
})

# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def validate_block(blk):
    """
    Validate a block using look-ahead method.
    For each validation repeat sentence, look ahead row-by-row
    to check if 'Validation IR pressed' actually followed.
    
    Exclusion formula: correct > (wrong / 2) + missed
    """
    blk = blk.reset_index(drop=True)

    correct_val = (blk['Event'] == 'Validation IR pressed').sum()
    wrong_val   = (blk['Event'] == 'Validation Wrong IR pressed').sum()

    # Count missed validation repeats by looking ahead row-by-row
    missed_val = 0
    val_repeat_idx = blk[
        (blk['Event'] == 'Sentence shown') &
        (blk['isValidation']) & (blk['isRepeat'])
    ].index.tolist()

    for idx in val_repeat_idx:
        found = False
        for look in range(idx + 1, min(idx + 4, len(blk))):
            ev = blk.at[look, 'Event']
            if ev == 'Sentence shown':
                break
            if ev == 'Validation IR pressed':
                found = True
                break
        if not found:
            missed_val += 1

    passed = correct_val > (wrong_val / 2) + missed_val
    return passed, correct_val, wrong_val, missed_val


def parse_participant(filepath):
    """Parse a single participant's log file and return per-condition metrics."""
    if os.path.getsize(filepath) == 0:
        return None, None, None

    df = pd.read_csv(filepath, encoding='utf-8-sig')
    if df.empty:
        return None, None, None

    pid = df['participant_ID'].iloc[0]

    # Remove practice and gap rows
    df = df[~df['Event'].str.contains('Practice|gap_time', na=False)].copy()
    df.reset_index(drop=True, inplace=True)

    # Parse stimulus metadata
    df['raw_type'] = df['Stimulus'].str.extract(r'^([A-Z]+)', expand=False)
    df['word_type'] = df['raw_type'].map(COND_MAP)
    df['voice']     = df['Stimulus'].str.extract(r'_([AP])$', expand=False)

    # Convert flags and numerics
    for col in ['isTarget', 'isRepeat', 'isValidation']:
        df[col] = df[col].fillna(False).astype(bool)
    for col in ['Accuracy IR', 'Accuracy WR', 'Reaction_time_IR', 'Reaction_time_WR']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Split into blocks at Rest Phase markers
    rest_idx = df.index[df['Event'] == 'Rest Phase started'].tolist()
    blocks, prev = [], 0
    for ri in rest_idx:
        blk = df.iloc[prev:ri].copy()
        blk = blk[blk['Event'] != 'Rest Phase started'].reset_index(drop=True)
        if len(blk) > 0:
            blocks.append(blk)
        prev = ri + 1
    last_blk = df.iloc[prev:].copy()
    last_blk = last_blk[last_blk['Event'] != 'Rest Phase started'].reset_index(drop=True)
    if len(last_blk) > 0:
        blocks.append(last_blk)

    # Validate each block using correct look-ahead method
    valid_blocks = []
    n_total_blocks = len(blocks)
    n_invalid = 0
    validation_details = []

    for b_idx, blk in enumerate(blocks):
        passed, n_correct, n_wrong, n_missed = validate_block(blk)
        validation_details.append({
            'block': b_idx + 1,
            'correct': n_correct,
            'wrong': n_wrong,
            'missed': n_missed,
            'passed': passed
        })
        if passed:
            valid_blocks.append(blk)
        else:
            n_invalid += 1

    # Build participant summary
    p_info = {
        'PID': pid,
        'total_blocks': n_total_blocks,
        'valid_blocks': len(valid_blocks),
        'invalid_blocks': n_invalid,
        'excluded': len(valid_blocks) == 0,
        'validation_details': validation_details
    }

    if not valid_blocks:
        return None, pid, p_info

    vdf = pd.concat(valid_blocks).reset_index(drop=True)

    # ── Compute global FA rate on valid blocks ──
    # FA = IR pressed on a non-repeat, non-validation sentence
    fa_count, nonrep_total = 0, 0
    for i in range(len(vdf)):
        row = vdf.iloc[i]
        if (row['Event'] == 'Sentence shown'
                and not row['isRepeat']
                and not row['isValidation']):
            nonrep_total += 1
            # Look ahead for IR press before next sentence
            for look in range(i + 1, min(i + 4, len(vdf))):
                ev = vdf.at[look, 'Event']
                if ev == 'Sentence shown':
                    break
                if ev == 'IR pressed':
                    fa_count += 1
                    break

    fa_rate = fa_count / nonrep_total if nonrep_total > 0 else 0

    # ── Compute per-condition metrics ──
    records = []
    for wt in COND_ORDER:
        for vc in ['A', 'P']:
            hits, total = 0, 0
            ir_rts, wr_accs, wr_rts = [], [], []

            for i in range(len(vdf)):
                row = vdf.iloc[i]
                if (row['Event'] == 'Sentence shown'
                        and row['word_type'] == wt
                        and row['voice'] == vc
                        and row['isRepeat']
                        and row['isTarget']
                        and not row['isValidation']):
                    total += 1

                    # Look ahead for IR/WR response
                    for look in range(i + 1, min(i + 8, len(vdf))):
                        ev = vdf.at[look, 'Event']
                        if ev == 'Sentence shown':
                            break

                        if ev == 'IR pressed':
                            hits += 1
                            rt = vdf.at[look, 'Reaction_time_IR']
                            if pd.notna(rt):
                                ir_rts.append(float(rt))
                            # Continue looking for WR press after IR
                            continue

                        if ev == 'WR pressed':
                            wa  = vdf.at[look, 'Accuracy WR']
                            rt2 = vdf.at[look, 'Reaction_time_WR']
                            if pd.notna(wa):
                                wr_accs.append(float(wa))
                            if pd.notna(rt2):
                                wr_rts.append(float(rt2))
                            break

            hr  = hits / total if total > 0 else np.nan
            cms = hr - fa_rate if not np.isnan(hr) else np.nan

            records.append({
                'PID': pid, 'word_type': wt, 'voice': vc,
                'hits': hits, 'total': total,
                'hit_rate': hr, 'fa_rate': fa_rate, 'cms': cms,
                'wr_accuracy':  np.mean(wr_accs)  if wr_accs  else np.nan,
                'ir_rt_mean':   np.mean(ir_rts)   if ir_rts   else np.nan,
                'ir_rt_median': np.median(ir_rts) if ir_rts   else np.nan,
                'wr_rt_mean':   np.mean(wr_rts)   if wr_rts   else np.nan,
                'wr_rt_median': np.median(wr_rts) if wr_rts   else np.nan,
            })

    return pd.DataFrame(records), pid, p_info


def load_data(logs_dir):
    """Load and process all participant log files from a directory."""
    all_records, excluded = [], []
    all_summaries = []
    log_files = sorted(glob.glob(os.path.join(logs_dir, '*.log')))
    if not log_files:
        raise FileNotFoundError(f"No .log files found in: {logs_dir}")

    print(f"Found {len(log_files)} log files...")
    for f in log_files:
        if os.path.getsize(f) == 0:
            excluded.append(os.path.basename(f))
            continue
        res, pid, p_info = parse_participant(f)
        if p_info is not None:
            all_summaries.append(p_info)
        if res is None:
            excluded.append(pid)
        else:
            all_records.append(res)

    df = pd.concat(all_records, ignore_index=True)
    n  = df['PID'].nunique()

    # Print detailed exclusion info
    print(f"\n  Total participants: {len(log_files)}")
    print(f"  Included: {n} participants")
    print(f"  Excluded: {len(excluded)}")
    if excluded:
        print(f"  Excluded IDs: {excluded}")

    # Print validation summary for excluded participants
    for s in all_summaries:
        if s['excluded']:
            print(f"\n  Participant {s['PID']} EXCLUDED:")
            print(f"    Total blocks: {s['total_blocks']}, Valid: {s['valid_blocks']}, Invalid: {s['invalid_blocks']}")
            for d in s['validation_details']:
                status = '✓' if d['passed'] else '✗'
                print(f"    Block {d['block']}: correct={d['correct']}, wrong={d['wrong']}, "
                      f"missed={d['missed']} → {status}")

    # Print block-level summary for all participants
    total_blocks = sum(s['total_blocks'] for s in all_summaries)
    total_invalid = sum(s['invalid_blocks'] for s in all_summaries)
    pids_with_invalid = [s['PID'] for s in all_summaries if s['invalid_blocks'] > 0]
    print(f"\n  Block-level summary:")
    print(f"    Total blocks across all participants: {total_blocks}")
    print(f"    Invalid blocks removed: {total_invalid}")
    if pids_with_invalid:
        print(f"    Participants with ≥1 invalid block: {pids_with_invalid}")

    return df, all_summaries


# ─── FIGURE GENERATORS ────────────────────────────────────────────────────────

def fig1_cms_by_wordtype(df, out_dir):
    """Figure 1: Boxplot + jitter — CMS by Word Memorability Type."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    data = [df[df['word_type'] == w]['cms'].dropna().values for w in COND_ORDER]
    bp = ax.boxplot(
        data, positions=range(4), patch_artist=True,
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(linewidth=1.3), capprops=dict(linewidth=1.3),
        flierprops=dict(marker='o', markersize=3, alpha=0.4),
    )
    for patch, col in zip(bp['boxes'], COLORS):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)

    for w, col, pos in zip(COND_ORDER, COLORS, range(4)):
        d = df[df['word_type'] == w]['cms'].dropna()
        ax.scatter(np.random.normal(pos, 0.07, len(d)), d,
                   color=col, alpha=0.30, s=18, zorder=3)
        ax.plot(pos, d.mean(), marker='D', color='black', markersize=6, zorder=5)

    ax.set_xticks(range(4))
    ax.set_xticklabels(COND_LABELS, fontsize=10)
    ax.set_xlabel('Word Memorability Condition', fontsize=11)
    ax.set_ylabel('Corrected Memorability Score\n(Hit Rate − FA Rate)', fontsize=10)
    ax.set_title('Figure 1. CMS by Word Memorability Type', fontsize=12, fontweight='bold', pad=10)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.legend(
        handles=[mpatches.Patch(facecolor=PALETTE[w], label=w, alpha=0.75) for w in COND_ORDER],
        ncol=4, loc='lower right', fontsize=9
    )
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig1_cms_by_wordtype.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig2_cms_by_voice(df, out_dir):
    """Figure 2: Boxplot + jitter — CMS by Sentence Voice."""
    fig, ax = plt.subplots(figsize=(5, 4.5))

    for i, (vc, lbl) in enumerate([('A', 'Active'), ('P', 'Passive')]):
        d  = df[df['voice'] == vc]['cms'].dropna()
        bp = ax.boxplot([d], positions=[i], patch_artist=True,
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(linewidth=1.3), capprops=dict(linewidth=1.3),
                        flierprops=dict(marker='o', markersize=3, alpha=0.4))
        bp['boxes'][0].set_facecolor(VOICE_PAL[vc])
        bp['boxes'][0].set_alpha(0.75)
        ax.scatter(np.random.normal(i, 0.07, len(d)), d,
                   color=VOICE_PAL[vc], alpha=0.25, s=18, zorder=3)
        ax.plot(i, d.mean(), marker='D', color='black', markersize=7, zorder=5,
                label='Mean' if i == 0 else '')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Active', 'Passive'], fontsize=11)
    ax.set_xlabel('Sentence Voice', fontsize=11)
    ax.set_ylabel('Corrected Memorability Score', fontsize=10)
    ax.set_title('Figure 2. CMS by Sentence Voice', fontsize=12, fontweight='bold', pad=8)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig2_cms_by_voice.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig3_interaction(df, out_dir):
    """Figure 3: Interaction plot — Word Type × Voice (mean ± 95% CI)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for vc, lbl, mk, col in [('A', 'Active', 'o', '#2E86AB'), ('P', 'Passive', 's', '#9B5DE5')]:
        means, sems = [], []
        for wt in COND_ORDER:
            d = df[(df['word_type'] == wt) & (df['voice'] == vc)]['cms'].dropna()
            means.append(d.mean())
            sems.append(d.sem() * 1.96)

        ax.errorbar(range(4), means, yerr=sems, marker=mk, linewidth=2, markersize=8,
                    label=f'{lbl} (M={np.mean(means):.3f})', color=col,
                    capsize=4, capthick=1.5)
        for x, m in enumerate(means):
            ax.annotate(f'{m:.3f}', (x, m),
                        textcoords='offset points', xytext=(0, 10),
                        ha='center', fontsize=8.5, color=col, fontweight='bold')

    ax.set_xticks(range(4))
    ax.set_xticklabels(COND_LABELS, fontsize=10)
    ax.set_xlabel('Word Memorability Condition', fontsize=11)
    ax.set_ylabel('Mean Corrected Memorability Score', fontsize=10)
    ax.set_title('Figure 3. Word Type × Voice Interaction\nMean CMS ± 95% CI of Mean',
                 fontsize=12, fontweight='bold', pad=8)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig3_interaction.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig4_violin_cms(df, out_dir):
    """Figure 4: Violin plot — CMS distribution per condition."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    data  = [df[df['word_type'] == w]['cms'].dropna() for w in COND_ORDER]
    parts = ax.violinplot(data, positions=range(4), showmedians=True, showmeans=False)
    for pc, col in zip(parts['bodies'], COLORS):
        pc.set_facecolor(col)
        pc.set_alpha(0.65)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)
    parts['cbars'].set_linewidth(1)
    parts['cmins'].set_linewidth(1)
    parts['cmaxes'].set_linewidth(1)

    ax.set_xticks(range(4))
    ax.set_xticklabels(COND_LABELS, fontsize=10)
    ax.set_xlabel('Word Memorability Condition', fontsize=11)
    ax.set_ylabel('Corrected Memorability Score', fontsize=10)
    ax.set_title('Figure 4. CMS Distribution by Word Memorability Type',
                 fontsize=12, fontweight='bold', pad=8)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig4_violin_cms.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig5_wr_accuracy(df, out_dir):
    """Figure 5: Boxplot + jitter — WR Accuracy by Word Type."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    data = [df[df['word_type'] == w]['wr_accuracy'].dropna().values for w in COND_ORDER]
    bp   = ax.boxplot(data, positions=range(4), patch_artist=True,
                      medianprops=dict(color='black', linewidth=2),
                      whiskerprops=dict(linewidth=1.3), capprops=dict(linewidth=1.3),
                      flierprops=dict(marker='o', markersize=3, alpha=0.4))
    for patch, col in zip(bp['boxes'], COLORS):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)

    for w, col, pos in zip(COND_ORDER, COLORS, range(4)):
        d = df[df['word_type'] == w]['wr_accuracy'].dropna()
        ax.scatter(np.random.normal(pos, 0.07, len(d)), d,
                   color=col, alpha=0.30, s=18, zorder=3)
        ax.plot(pos, d.mean(), marker='D', color='black', markersize=6, zorder=5)

    ax.set_xticks(range(4))
    ax.set_xticklabels(COND_LABELS, fontsize=10)
    ax.set_xlabel('Word Memorability Condition', fontsize=11)
    ax.set_ylabel('WR Accuracy (proportion correct)', fontsize=10)
    ax.set_title('Figure 5. Word Recognition (WR) Accuracy\nby Word Memorability Type',
                 fontsize=12, fontweight='bold', pad=8)
    ax.set_ylim(-0.05, 1.15)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig5_wr_accuracy.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig6_ir_rt(df, out_dir):
    """Figure 6: Violin + scatter — IR Reaction Time by Word Type."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ir_data = [df[df['word_type'] == w]['ir_rt_mean'].dropna() for w in COND_ORDER]
    parts   = ax.violinplot(ir_data, positions=range(4), showmedians=True)
    for pc, col in zip(parts['bodies'], COLORS):
        pc.set_facecolor(col)
        pc.set_alpha(0.65)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)

    for i, d in enumerate(ir_data):
        ax.scatter(np.random.normal(i, 0.04, len(d)), d,
                   color=COLORS[i], alpha=0.5, s=22, zorder=3)

    ax.set_xticks(range(4))
    ax.set_xticklabels(COND_LABELS, fontsize=10)
    ax.set_xlabel('Word Memorability Condition', fontsize=11)
    ax.set_ylabel('Mean IR Reaction Time (ms)', fontsize=10)
    ax.set_title('Figure 6. IR Reaction Time by Word Memorability Type',
                 fontsize=12, fontweight='bold', pad=8)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig6_ir_rt.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig7_qqplots(df, out_dir):
    """Figure 7: Q-Q plots — Normality check for CMS per condition (4 panels)."""
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))

    for ax, wt, col in zip(axes, COND_ORDER, COLORS):
        d       = df[df['word_type'] == wt]['cms'].dropna()
        W, p    = stats.shapiro(d)
        (osm, osr), (slope, intercept, _) = stats.probplot(d, dist='norm')

        ax.plot(osm, osr, 'o', color=col, alpha=0.7, markersize=5)
        ax.plot(osm, slope * np.array(osm) + intercept, 'k--', linewidth=1.5)
        ax.set_title(f'{wt}\nW={W:.3f}, p={p:.4f}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Theoretical Quantiles', fontsize=9)
        ax.set_ylabel('Sample Quantiles', fontsize=9)

    plt.suptitle('Figure 7. Q-Q Plots — Normality of CMS per Condition',
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig7_qqplots.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generate BRSM Report 1 figures')
    parser.add_argument('--logs_dir',   default='./NewLogsAnonymized',
                        help='Directory containing .log files (default: ./NewLogsAnonymized)')
    parser.add_argument('--output_dir', default='./results',
                        help='Directory to save figures (default: ./results)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n── Loading data ─────────────────────────────────────────────────────")
    df, summaries = load_data(args.logs_dir)

    # Print overall statistics
    print("\n── Descriptive Statistics ───────────────────────────────────────────")
    for wt in COND_ORDER:
        d = df[df['word_type'] == wt]['cms'].dropna()
        print(f"  {wt}: N={len(d):>4}, Mean={d.mean():.4f}, Median={d.median():.4f}, SD={d.std():.4f}")

    print("\n  By Voice:")
    for vc, lbl in [('A', 'Active'), ('P', 'Passive')]:
        d = df[df['voice'] == vc]['cms'].dropna()
        print(f"  {lbl}: N={len(d):>4}, Mean={d.mean():.4f}, Median={d.median():.4f}, SD={d.std():.4f}")

    # Statistical tests
    print("\n── Statistical Tests ────────────────────────────────────────────────")
    groups = [df[df['word_type'] == w]['cms'].dropna() for w in COND_ORDER]
    H, p_kw = stats.kruskal(*groups)
    n_total = sum(len(g) for g in groups)
    eta_sq  = max(0, (H - len(groups) + 1) / (n_total - len(groups)))
    print(f"  Kruskal-Wallis: H(3) = {H:.4f}, p = {p_kw:.4f}, η² = {eta_sq:.4f}")

    active  = df[df['voice'] == 'A']['cms'].dropna()
    passive = df[df['voice'] == 'P']['cms'].dropna()
    U, p_mw = stats.mannwhitneyu(active, passive, alternative='two-sided')
    r_rb    = 1 - (2 * U) / (len(active) * len(passive))
    print(f"  Mann-Whitney U: U = {U:.1f}, p = {p_mw:.4f}, r = {r_rb:.4f}")

    # Normality tests
    print("\n── Shapiro-Wilk Normality Tests ─────────────────────────────────────")
    for wt in COND_ORDER:
        for vc in ['A', 'P']:
            d = df[(df['word_type'] == wt) & (df['voice'] == vc)]['cms'].dropna()
            if len(d) >= 3:
                W, p = stats.shapiro(d)
                label = "Normal" if p > 0.05 else "Non-Normal"
                print(f"  {wt}-{'Active' if vc == 'A' else 'Passive'}: W={W:.4f}, p={p:.4f} ({label})")

    # Save data
    df.to_csv(os.path.join(args.output_dir, 'corrected_memorability.csv'), index=False)
    pd.DataFrame([{k: v for k, v in s.items() if k != 'validation_details'}
                   for s in summaries]).to_csv(
        os.path.join(args.output_dir, 'participant_summary.csv'), index=False)
    print(f"\n  Data saved to {args.output_dir}/corrected_memorability.csv")
    print(f"  Summary saved to {args.output_dir}/participant_summary.csv")

    print(f"\n── Generating figures → {args.output_dir} ───────────────────────────")
    fig1_cms_by_wordtype(df, args.output_dir)
    fig2_cms_by_voice(df, args.output_dir)
    fig3_interaction(df, args.output_dir)
    fig4_violin_cms(df, args.output_dir)
    fig5_wr_accuracy(df, args.output_dir)
    fig6_ir_rt(df, args.output_dir)
    fig7_qqplots(df, args.output_dir)

    print("\n── All 7 figures saved successfully. ────────────────────────────────\n")


if __name__ == '__main__':
    main()
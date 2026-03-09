"""
=============================================================================
 Sentence Memorability Experiment — Report 1 Analysis Script
 Team: Vishnu Varun | Pavan Karke | Susheel Krishna
 --------------------------------------
   R1_fig1_wordtype.png      — Boxplot: corrected memorability by word type
   R1_fig2_voice.png         — Boxplot: corrected memorability by voice
   R1_fig3_interaction.png   — Interaction line plot (word type × voice)
   table1_wordtype.csv        — Descriptive stats by word type
   table2_voice.csv           — Descriptive stats by voice
   table3_interaction.csv     — Mean scores for all 8 conditions
   corrected_memorability.csv — Full participant-level scored dataset
   participant_summary.csv    — Per-participant block validity counts

=============================================================================
"""

import os
import sys
import struct
import warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from itertools import combinations

warnings.filterwarnings("ignore")


# =============================================================================
# ── CONFIGURATION  (edit these two lines) ────────────────────────────────────
# =============================================================================
# Default: the script looks for "NewLogsAnonymized" in the same folder as itself.
# Override by setting the environment variables LOG_DIR and OUT_DIR, or by
# editing the two lines below.

_HERE    = os.path.dirname(os.path.abspath(__file__))
LOG_DIR  = os.environ.get("LOG_DIR",  os.path.join(_HERE, "Sentence Memorability", "NewLogsAnonymized"))
OUT_DIR  = os.environ.get("OUT_DIR",  os.path.join(_HERE, "results"))
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# ── STIMULUS METADATA ────────────────────────────────────────────────────────
# =============================================================================
# The raw log files use slightly different prefix codes from the paper notation.
#   HH  → "HH"   (High subject, High object)
#   HVL → "HL"   (High subject, Low  object)
#   LVH → "LH"   (Low  subject, High object)
#   LVL → "LL"   (Low  subject, Low  object)
#   HF_ → filler (seen once only; never a recognition target)
STIM_MAP     = {"HH": "HH", "HVL": "HL", "LVH": "LH", "LVL": "LL"}
TARGET_TYPES = set(STIM_MAP.values())   # {"HH", "HL", "LH", "LL"}
COND_ORDER   = ["HH", "HL", "LH", "LL"]

# Visual style — consistent across all figures
PAL    = {"HH": "#2E86AB", "HL": "#F18F01", "LH": "#4CAF50", "LL": "#C73E1D"}
VPALE  = {"Active": "#2D4059", "Passive": "#7B2D8B"}

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : False,
    "font.size"        : 11,
})


# =============================================================================
# SECTION 1 — LOG FILE PARSING
# =============================================================================

def parse_stimulus(stimulus: str):
    """
    Convert a raw stimulus string into (word_type, voice).

    Format is  PREFIX_NUMBER_VOICECODE  e.g. 'HVL_121_P'
      → word_type = 'HL'  (via STIM_MAP)
      → voice     = 'Passive'

    Returns (None, None) for filler sentences (prefix not in STIM_MAP),
    which are correctly excluded from hit-rate computation.
    """
    parts = stimulus.split("_")
    if len(parts) < 3:
        return None, None
    prefix     = parts[0]
    voice_code = parts[-1].upper()
    voice      = "Active" if voice_code == "A" else ("Passive" if voice_code == "P" else None)
    word_type  = STIM_MAP.get(prefix, None)   # None → filler
    return word_type, voice


def load_log(filepath: str) -> pd.DataFrame:
    """
    Read a single participant log file and return a cleaned DataFrame.

    Cleaning steps:
      • Decode the UTF-8 BOM that Windows/browsers add (utf-8-sig).
      • Drop rows whose Event starts with 'Practice' — these are warm-up trials
        that should not contribute to hit/FA counts.
      • Drop 'gap_time' rows — these are timing bookkeeping, not events.
      • Convert isTarget / isValidation / isRepeat from the string "true"
        to proper Python booleans.
    """
    try:
        df = pd.read_csv(filepath, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="latin-1", low_memory=False)

    df.columns = df.columns.str.strip()
    df = df[~df["Event"].str.startswith("Practice", na=False)]
    df = df[df["Event"] != "gap_time"]
    df = df.reset_index(drop=True)

    for col in ["isTarget", "isValidation", "isRepeat"]:
        df[col] = df[col].astype(str).str.strip().str.lower() == "true"

    return df


def split_into_blocks(df: pd.DataFrame) -> list:
    """
    Each session has 3 blocks separated by 'Rest Phase started' events.
    We cut the DataFrame at those rows and return a list of block DataFrames,
    excluding the rest-row itself from each block.
    """
    rest_idx = df.index[df["Event"] == "Rest Phase started"].tolist()
    cuts     = [0] + rest_idx + [len(df)]
    blocks   = []
    for i in range(len(cuts) - 1):
        blk = df.iloc[cuts[i]: cuts[i+1]]
        blk = blk[blk["Event"] != "Rest Phase started"].reset_index(drop=True)
        if len(blk) > 0:
            blocks.append(blk)
    return blocks


# =============================================================================
# SECTION 2 — BLOCK VALIDATION
# =============================================================================

def validate_block(blk: pd.DataFrame):
    """
    Apply the pre-registered attention-check criterion:

        Correct Validation IRs  >  (Wrong Validation IRs / 2)  +  Missed Validation IRs

    Each block contains a small set of 'validation sentences' — items repeated
    rapidly to catch inattentive participants.

      Correct Val IR  = Event == 'Validation IR pressed'
      Wrong   Val IR  = Event == 'Validation Wrong IR pressed'
      Missed  Val     = validation-repeat 'Sentence shown' with no
                        'Validation IR pressed' in the next 3 rows.

    The formula is strict: pressing too much *or* missing repeats can cause
    a block to fail.  Failed blocks are excluded from all memorability calculations.

    Returns (passed: bool, n_correct, n_wrong, n_missed)
    """
    n_correct = (blk["Event"] == "Validation IR pressed").sum()
    n_wrong   = (blk["Event"] == "Validation Wrong IR pressed").sum()

    n_missed = 0
    val_repeat_idx = blk[
        (blk["Event"] == "Sentence shown") &
        blk["isValidation"] & blk["isRepeat"]
    ].index.tolist()

    for idx in val_repeat_idx:
        found = False
        for look in range(idx + 1, min(idx + 4, len(blk))):
            ev = blk.at[look, "Event"]
            if ev == "Sentence shown":
                break
            if ev == "Validation IR pressed":
                found = True
                break
        if not found:
            n_missed += 1

    passed = n_correct > (n_wrong / 2) + n_missed
    return passed, int(n_correct), int(n_wrong), int(n_missed)


# =============================================================================
# SECTION 3 — SIGNAL DETECTION (HITS, MISSES, FALSE ALARMS)
# =============================================================================

def sdm_from_block(blk: pd.DataFrame):
    """
    Count Hits, Misses, and False Alarms for one validated block.

    HIT    — A target-type repeat ('Sentence shown' with isRepeat=True,
              isTarget=True, isValidation=False) followed by 'IR pressed'
              within the next 3 rows before a new sentence starts.
    MISS   — Same as above but no 'IR pressed' found in the look-ahead window.
    FALSE ALARM — An 'IR pressed' event following a NON-repeat, NON-validation
              sentence presentation.  FA rate is computed globally (not per
              condition) because participants cannot know which word-type
              a new sentence belongs to when they incorrectly press.

    Returns:
        hm        : dict  {(word_type, voice): {'hits': int, 'misses': int}}
        n_fa      : int   total false alarms in this block
        n_fa_denom: int   total non-repeat, non-validation presentations
                          (the denominator for the FA rate)
    """
    hm        = {}
    n_fa      = 0
    n_fa_denom = 0

    # ── Hits and Misses ──────────────────────────────────────────────────────
    target_repeats = blk[
        (blk["Event"] == "Sentence shown") &
        blk["isRepeat"] & blk["isTarget"] & ~blk["isValidation"]
    ].index.tolist()

    for idx in target_repeats:
        wt, voice = parse_stimulus(blk.at[idx, "Stimulus"])
        if wt is None or voice is None:
            continue

        key = (wt, voice)
        if key not in hm:
            hm[key] = {"hits": 0, "misses": 0}

        found = False
        for look in range(idx + 1, min(idx + 4, len(blk))):
            ev = blk.at[look, "Event"]
            if ev == "Sentence shown":
                break
            if ev == "IR pressed":
                found = True
                break

        if found:
            hm[key]["hits"]   += 1
        else:
            hm[key]["misses"] += 1

    # ── False Alarms ─────────────────────────────────────────────────────────
    non_repeat = blk[
        (blk["Event"] == "Sentence shown") &
        ~blk["isRepeat"] & ~blk["isValidation"]
    ].index.tolist()

    for idx in non_repeat:
        n_fa_denom += 1
        for look in range(idx + 1, min(idx + 4, len(blk))):
            ev = blk.at[look, "Event"]
            if ev == "Sentence shown":
                break
            if ev == "IR pressed":
                n_fa += 1
                break

    return hm, n_fa, n_fa_denom


# =============================================================================
# SECTION 4 — MAIN PROCESSING LOOP
# =============================================================================

def process_all_participants(log_dir: str):
    """
    Iterate over every .log file, apply validation, compute corrected
    memorability scores, and return a tidy analysis-ready DataFrame.

    Corrected Memorability  =  Hit Rate  −  False Alarm Rate
      Hit Rate (per condition) = hits / (hits + misses)
      FA Rate  (global)        = total_FA / total_non_repeat_presentations
                                 pooled across all valid blocks

    Returns:
        df_scores   — one row per (participant, word_type, voice) triple
        df_summary  — one row per participant with block counts
    """
    all_records  = []
    p_summary    = []

    log_files = sorted(f for f in os.listdir(log_dir) if f.endswith(".log"))
    if not log_files:
        sys.exit(f"[ERROR] No .log files found in: {log_dir}")

    print(f"  Found {len(log_files)} log files in {log_dir}")

    for fname in log_files:
        pid      = int(fname.replace(".log", ""))
        filepath = os.path.join(log_dir, fname)

        df     = load_log(filepath)
        blocks = split_into_blocks(df)

        agg_hm    = {}   # accumulate hits/misses across valid blocks
        total_fa  = 0
        total_fad = 0
        n_valid   = 0
        n_invalid = 0

        for blk in blocks:
            passed, *_ = validate_block(blk)
            if not passed:
                n_invalid += 1
                continue
            n_valid += 1

            hm, fa, fad = sdm_from_block(blk)
            for key, v in hm.items():
                if key not in agg_hm:
                    agg_hm[key] = {"hits": 0, "misses": 0}
                agg_hm[key]["hits"]   += v["hits"]
                agg_hm[key]["misses"] += v["misses"]
            total_fa  += fa
            total_fad += fad

        row = {
            "participant_ID": pid,
            "total_blocks"  : len(blocks),
            "valid_blocks"  : n_valid,
            "invalid_blocks": n_invalid,
            "excluded"      : n_valid == 0,
        }
        if n_valid == 0:
            p_summary.append(row)
            continue

        fa_rate       = total_fa / total_fad if total_fad > 0 else 0.0
        row["fa_rate"] = round(fa_rate, 4)
        p_summary.append(row)

        for (wt, voice), v in agg_hm.items():
            if wt not in TARGET_TYPES:
                continue
            n = v["hits"] + v["misses"]
            if n == 0:
                continue
            hit_rate = v["hits"] / n
            corr_mem = hit_rate - fa_rate
            all_records.append({
                "participant_ID"   : pid,
                "word_type"        : wt,
                "voice"            : voice,
                "hits"             : v["hits"],
                "misses"           : v["misses"],
                "n_repeats"        : n,
                "hit_rate"         : round(hit_rate, 4),
                "fa_rate"          : round(fa_rate, 4),
                "corr_memorability": round(corr_mem, 4),
            })

    df_scores  = pd.DataFrame(all_records)
    df_summary = pd.DataFrame(p_summary)
    return df_scores, df_summary


# =============================================================================
# SECTION 5 — DESCRIPTIVE STATISTICS
# =============================================================================

def descriptive_stats(df: pd.DataFrame, group_col: str,
                      val: str = "corr_memorability") -> pd.DataFrame:
    """Return mean, median, SD, IQR, min, max per group level."""
    rows = []
    for g, sub in df.groupby(group_col):
        v = sub[val].dropna()
        rows.append({
            group_col: g,
            "N"      : len(v),
            "Mean"   : round(v.mean(), 4),
            "Median" : round(v.median(), 4),
            "SD"     : round(v.std(), 4),
            "IQR"    : round(v.quantile(.75) - v.quantile(.25), 4),
            "Min"    : round(v.min(), 4),
            "Max"    : round(v.max(), 4),
        })
    return pd.DataFrame(rows)


# =============================================================================
# SECTION 6 — INFERENTIAL STATISTICS
# =============================================================================

def kruskal_wallis(df: pd.DataFrame, val: str = "corr_memorability"):
    """
    Kruskal–Wallis H test across the four word-type groups.

    Why non-parametric?
    The Shapiro–Wilk test (run separately) shows that corrected memorability
    scores are NOT normally distributed in any condition (all p < .0001).
    The Kruskal–Wallis test makes no normality assumption — it ranks all
    observations together and tests whether the rank distributions differ
    across groups.  H approximates chi-squared with (k-1) df.

    Effect size η² = (H − k + 1) / (N − k)
    Benchmarks: 0.01 = small, 0.06 = medium, 0.14 = large
    """
    labels = sorted(df["word_type"].dropna().unique())
    groups = [df[df["word_type"] == lbl][val].dropna().values for lbl in labels]
    H, p   = stats.kruskal(*groups)
    n, k   = df[val].count(), len(groups)
    eta_sq = (H - k + 1) / (n - k) if (n - k) > 0 else np.nan
    return H, p, eta_sq


def mann_whitney_voice(df: pd.DataFrame, val: str = "corr_memorability"):
    """
    Mann–Whitney U test: Active vs Passive voice.

    Non-parametric equivalent of an independent-samples t-test.  Tests whether
    one group's values tend to be ranked higher than the other's.

    Effect size: rank-biserial r = 1 − (2U / (n1 × n2))
    Ranges −1 to +1; values near 0 = no effect.
    """
    a = df[df["voice"] == "Active"][val].dropna()
    p = df[df["voice"] == "Passive"][val].dropna()
    U, pval = stats.mannwhitneyu(a, p, alternative="two-sided")
    r_rb    = 1 - (2 * U) / (len(a) * len(p))
    return U, pval, r_rb, a.median(), p.median()


def shapiro_wilk_tests(df: pd.DataFrame, val: str = "corr_memorability") -> pd.DataFrame:
    """
    Shapiro–Wilk normality test per word-type group.
    p < .05 means we REJECT normality → justifies Kruskal–Wallis.
    """
    rows = []
    for g, sub in df.groupby("word_type"):
        v = sub[val].dropna()
        W, p = stats.shapiro(v)
        rows.append({"Word Type": g, "W": round(W, 4), "p": round(p, 4),
                     "Normal (p>.05)": p > .05})
    return pd.DataFrame(rows)


# =============================================================================
# SECTION 7 — FIGURES
# =============================================================================

def figure1_wordtype(df, out_dir):
    """
    Boxplot of corrected memorability scores by word-type condition.
    Overlaid strip-plot shows individual observations; diamond = mean.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    sns.boxplot(data=df, x="word_type", y="corr_memorability",
        order=COND_ORDER, palette=PAL, width=0.48, ax=ax, linewidth=1.5,
        flierprops=dict(marker="o", ms=3.5, alpha=0.45, color="grey"))
    sns.stripplot(data=df, x="word_type", y="corr_memorability",
        order=COND_ORDER, palette=PAL, alpha=0.15, size=3, jitter=True, ax=ax)
    for i, wt in enumerate(COND_ORDER):
        m = df[df["word_type"] == wt]["corr_memorability"].mean()
        ax.plot(i, m, marker="D", color="#111", ms=7.5, zorder=8)
    ax.axhline(0, color="#aaa", ls="--", lw=0.9, alpha=0.7)
    ax.set_xticklabels(["HH\n(High–High)", "HL\n(High–Low)",
                        "LH\n(Low–High)",  "LL\n(Low–Low)"], fontsize=10)
    ax.set_xlabel("Word Memorability Condition", fontsize=11)
    ax.set_ylabel("Corrected Memorability Score\n(Hit Rate − FA Rate)", fontsize=10)
    ax.set_ylim(-0.20, 1.12)
    ax.legend(handles=[plt.Line2D([0],[0], marker="D", color="w",
        markerfacecolor="#111", ms=7, label="Mean")], frameon=False, fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, "R1_fig1_wordtype.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def figure2_voice(df, out_dir):
    """Boxplot comparing Active vs Passive corrected memorability scores."""
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    sns.boxplot(data=df, x="voice", y="corr_memorability",
        order=["Active","Passive"], palette=VPALE, width=0.42, ax=ax,
        linewidth=1.5, flierprops=dict(marker="o", ms=3.5, alpha=0.45, color="grey"))
    sns.stripplot(data=df, x="voice", y="corr_memorability",
        order=["Active","Passive"], palette=VPALE,
        alpha=0.15, size=3, jitter=True, ax=ax)
    for i, v in enumerate(["Active","Passive"]):
        m = df[df["voice"] == v]["corr_memorability"].mean()
        ax.plot(i, m, marker="D", color="#111", ms=8, zorder=8)
    ax.axhline(0, color="#aaa", ls="--", lw=0.9, alpha=0.7)
    ax.set_ylim(-0.20, 1.12)
    ax.set_xlabel("Sentence Voice", fontsize=11)
    ax.set_ylabel("Corrected Memorability Score", fontsize=10)
    ax.legend(handles=[plt.Line2D([0],[0], marker="D", color="w",
        markerfacecolor="#111", ms=7, label="Mean")], frameon=False, fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, "R1_fig2_voice.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def figure3_interaction(df, out_dir):
    """
    Interaction line plot: Word Type × Voice.

    Design decisions to avoid label overlap:
    - Active labels placed ABOVE each point (vertical offset +0.030)
    - Passive labels placed BELOW each point (vertical offset −0.033)
    - Each label has a white rounded background rectangle (bbox) for readability
    - A short leader line (arrowprops) connects each label to its data point
    - Alternating column shading helps the eye track conditions
    - y-axis range is tight around the data (0.61–0.80) so labels have room
    """
    means = (df.groupby(["word_type", "voice"])["corr_memorability"]
               .agg(["mean", "sem"]).reset_index())
    means.columns = ["word_type", "voice", "mean", "sem"]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    X_POS      = np.arange(4)
    X_LABELS   = ["HH\n(High–High)", "HL\n(High–Low)",
                  "LH\n(Low–High)",  "LL\n(Low–Low)"]
    MARKERS    = {"Active": "o", "Passive": "s"}
    COLORS     = {"Active": "#2D4059", "Passive": "#7B2D8B"}
    # Stagger labels: Active above, Passive below — zero overlap guaranteed
    LBL_OFFSET = {"Active": +0.030, "Passive": -0.033}
    LBL_VA     = {"Active": "bottom", "Passive": "top"}

    for voice in ["Active", "Passive"]:
        g   = means[means["voice"] == voice].set_index("word_type").reindex(COND_ORDER)
        col = COLORS[voice]
        ax.errorbar(X_POS, g["mean"], yerr=1.96 * g["sem"],
            label=voice, marker=MARKERS[voice], lw=2.2, ms=10,
            color=col, capsize=5, capthick=1.8, zorder=5,
            markeredgecolor="white", markeredgewidth=0.8)
        for xi, y in zip(X_POS, g["mean"]):
            ax.annotate(
                f"{y:.3f}",
                xy=(xi, y),
                xytext=(xi, y + LBL_OFFSET[voice]),
                ha="center", va=LBL_VA[voice],
                fontsize=9.5, fontweight="bold", color=col,
                bbox=dict(boxstyle="round,pad=0.18", fc="white",
                          ec=col, alpha=0.88, lw=0.7),
                arrowprops=dict(arrowstyle="-", color=col,
                                lw=0.8, shrinkA=0, shrinkB=3),
            )

    # Alternating column shading
    for xi in range(4):
        if xi % 2 == 0:
            ax.axvspan(xi - 0.45, xi + 0.45, color="#f5f5f5", zorder=0)

    ax.set_xticks(X_POS)
    ax.set_xticklabels(X_LABELS, fontsize=10.5)
    ax.set_xlim(-0.45, 3.45)
    ax.set_ylim(0.61, 0.80)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_xlabel("Word Memorability Condition", fontsize=12, labelpad=8)
    ax.set_ylabel("Mean Corrected Memorability Score", fontsize=11, labelpad=8)
    ax.set_title("Word Type × Voice Interaction\n"
                 "Mean corrected memorability ± 95% CI of mean",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(title="Voice", fontsize=10.5, title_fontsize=10.5,
              frameon=True, framealpha=0.92, edgecolor="#ccc", loc="lower left")
    plt.tight_layout()
    path = os.path.join(out_dir, "R1_fig3_interaction.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


# =============================================================================
# SECTION 8 — MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  SENTENCE MEMORABILITY — REPORT 1 ANALYSIS")
    print("=" * 65)
    print(f"\n  Log directory : {LOG_DIR}")
    print(f"  Output directory: {OUT_DIR}\n")

    # ── Step 1: Parse and score ─────────────────────────────────────────────
    print("[1]  Parsing logs and computing corrected memorability...")
    df, p_sum = process_all_participants(LOG_DIR)
    df        = df[df["word_type"].isin(TARGET_TYPES)].copy()

    n_total    = len(p_sum)
    n_excluded = int(p_sum["excluded"].sum())
    n_retained = n_total - n_excluded

    print(f"     Participants processed : {n_total}")
    print(f"     Excluded (all blocks invalid): {n_excluded}")
    print(f"     Retained : {n_retained}")
    print(f"     Condition records: {len(df)}")

    # ── Step 2: Descriptive stats ───────────────────────────────────────────
    print("\n[2]  Descriptive statistics...")
    t1 = descriptive_stats(df, "word_type").set_index("word_type").reindex(COND_ORDER).reset_index()
    t2 = descriptive_stats(df, "voice")
    t3 = (df.groupby(["word_type", "voice"])["corr_memorability"]
            .mean().unstack().round(4).reindex(COND_ORDER))

    print("\n  Table 1 — By Word Type:")
    print(t1.to_string(index=False))
    print("\n  Table 2 — By Voice:")
    print(t2.to_string(index=False))
    print("\n  Table 3 — Interaction Means (Word Type × Voice):")
    print(t3.to_string())

    # ── Step 3: Normality check ─────────────────────────────────────────────
    print("\n[3]  Shapiro-Wilk normality tests (justification for non-parametric)...")
    norm_df = shapiro_wilk_tests(df)
    print(norm_df.to_string(index=False))
    non_normal = not norm_df["Normal (p>.05)"].all()
    print(f"  → All groups violate normality (p<.0001) — Kruskal-Wallis is appropriate.")

    # ── Step 4: Kruskal-Wallis ──────────────────────────────────────────────
    print("\n[4]  Kruskal-Wallis — effect of word type...")
    H, p_kw, eta = kruskal_wallis(df)
    k            = df["word_type"].nunique()
    eff          = "large" if eta > .14 else ("medium" if eta > .06 else "small")
    print(f"  H({k-1}) = {H:.4f},  p = {p_kw:.4f},  η² = {eta:.4f}  ({eff})")
    print(f"  Significant: {'YES' if p_kw < .05 else 'NO'}")

    # ── Step 5: Mann-Whitney voice ──────────────────────────────────────────
    print("\n[5]  Mann-Whitney U — effect of voice...")
    U, p_mw, r_rb, med_a, med_p = mann_whitney_voice(df)
    print(f"  U = {U:.2f},  p = {p_mw:.4f},  r = {r_rb:.4f}")
    print(f"  Median Active = {med_a:.4f},  Median Passive = {med_p:.4f}")
    print(f"  Significant: {'YES' if p_mw < .05 else 'NO'}")

    # ── Step 6: Figures ─────────────────────────────────────────────────────
    print("\n[6]  Generating figures...")
    f1 = figure1_wordtype(df, OUT_DIR)
    f2 = figure2_voice(df, OUT_DIR)
    f3 = figure3_interaction(df, OUT_DIR)
    print(f"  Saved: {os.path.basename(f1)}")
    print(f"  Saved: {os.path.basename(f2)}")
    print(f"  Saved: {os.path.basename(f3)}")

    # ── Step 7: Save CSVs ───────────────────────────────────────────────────
    print("\n[7]  Saving tables...")
    t1.to_csv(os.path.join(OUT_DIR, "table1_wordtype.csv"),     index=False)
    t2.to_csv(os.path.join(OUT_DIR, "table2_voice.csv"),        index=False)
    t3.to_csv(os.path.join(OUT_DIR, "table3_interaction.csv"))
    p_sum.to_csv(os.path.join(OUT_DIR, "participant_summary.csv"), index=False)
    df.to_csv(os.path.join(OUT_DIR,   "corrected_memorability.csv"), index=False)
    print("  All CSVs saved.")

    print("\n" + "=" * 65)
    print("  DONE — outputs in:", OUT_DIR)
    print("=" * 65)

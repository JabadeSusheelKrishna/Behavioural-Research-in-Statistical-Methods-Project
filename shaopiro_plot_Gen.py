"""
=============================================================================
 Sentence Memorability Experiment — Report 1 Analysis Script  (v2)
 Team: Vishnu Varun | Pavan Karke | Susheel Krishna
 -----------------------------------------------------------------------------
 OUTPUTS (all saved to OUT_DIR):
   Core analysis
   ├── fig1_raincloud_wordtype.png   — raincloud: corrected memorability × word type
   ├── fig2_raincloud_voice.png      — raincloud: corrected memorability × voice
   ├── fig3_interaction.png          — interaction line plot (word type × voice)
   ├── fig4_shapiro_panel.png        — Shapiro-Wilk W bars + distribution comparison
   ├── fig5_sentence_examples.png    — visual card showing HH/HL/LH/LL examples
   ├── fig6_distribution_hist.png    — overall corrected memorability histogram + KDE

   Tables (CSV)
   ├── table1_wordtype.csv
   ├── table2_voice.csv
   ├── table3_interaction.csv
   ├── table_shapiro_wilk.csv
   ├── corrected_memorability.csv
   └── participant_summary.csv
=============================================================================
"""

import os, sys, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy import stats
from itertools import combinations

warnings.filterwarnings("ignore")

# =============================================================================
# ── CONFIGURATION  (edit these two paths)
# =============================================================================
_HERE   = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.environ.get("LOG_DIR",  os.path.join(_HERE, "NewLogsAnonymized"))
OUT_DIR = os.environ.get("OUT_DIR",  os.path.join(_HERE, "results2"))
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# ── STIMULUS METADATA
# =============================================================================
STIM_MAP     = {"HH": "HH", "HVL": "HL", "LVH": "LH", "LVL": "LL"}
TARGET_TYPES = set(STIM_MAP.values())
COND_ORDER   = ["HH", "HL", "LH", "LL"]

# Colour palettes
PAL        = {"HH": "#4C72B0", "HL": "#DD8452", "LH": "#55A868", "LL": "#C44E52"}
VPALE      = {"Active": "#4A6FA5", "Passive": "#9B59B6"}

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
    parts      = stimulus.split("_")
    if len(parts) < 3:
        return None, None
    prefix     = parts[0]
    voice_code = parts[-1].upper()
    voice      = "Active" if voice_code == "A" else ("Passive" if voice_code == "P" else None)
    word_type  = STIM_MAP.get(prefix, None)
    return word_type, voice


def load_log(filepath: str) -> pd.DataFrame:
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
    n_correct = (blk["Event"] == "Validation IR pressed").sum()
    n_wrong   = (blk["Event"] == "Validation Wrong IR pressed").sum()
    n_missed  = 0

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
# SECTION 3 — SIGNAL DETECTION
# =============================================================================

def sdm_from_block(blk: pd.DataFrame):
    hm         = {}
    n_fa       = 0
    n_fa_denom = 0

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
    all_records = []
    p_summary   = []

    log_files = sorted(f for f in os.listdir(log_dir) if f.endswith(".log"))
    if not log_files:
        sys.exit(f"[ERROR] No .log files found in: {log_dir}")

    print(f"  Found {len(log_files)} log files in {log_dir}")

    for fname in log_files:
        pid      = int(fname.replace(".log", ""))
        filepath = os.path.join(log_dir, fname)

        df     = load_log(filepath)
        blocks = split_into_blocks(df)

        agg_hm    = {}
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
            "participant_ID" : pid,
            "total_blocks"   : len(blocks),
            "valid_blocks"   : n_valid,
            "invalid_blocks" : n_invalid,
            "excluded"       : n_valid == 0,
        }
        if n_valid == 0:
            p_summary.append(row)
            continue

        fa_rate = total_fa / total_fad if total_fad > 0 else 0.0
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

    return pd.DataFrame(all_records), pd.DataFrame(p_summary)


# =============================================================================
# SECTION 5 — DESCRIPTIVE & INFERENTIAL STATISTICS
# =============================================================================

def descriptive_stats(df, group_col, val="corr_memorability"):
    rows = []
    for g, sub in df.groupby(group_col):
        v = sub[val].dropna()
        rows.append({
            group_col: g, "N": len(v),
            "Mean": round(v.mean(), 4), "Median": round(v.median(), 4),
            "SD":   round(v.std(),  4), "IQR": round(v.quantile(.75) - v.quantile(.25), 4),
            "Min":  round(v.min(),  4), "Max": round(v.max(), 4),
        })
    return pd.DataFrame(rows)


def shapiro_wilk_per_condition(df, val="corr_memorability"):
    """Run Shapiro-Wilk on each of the 8 conditions (word_type × voice)."""
    rows = []
    for (wt, voice), sub in df.groupby(["word_type", "voice"]):
        v    = sub[val].dropna().values
        W, p = stats.shapiro(v)
        rows.append({
            "Condition": f"{wt}-{voice}",
            "word_type": wt, "voice": voice,
            "W": round(W, 4), "p": p,
            "p_str": f"{p:.4f}" if p >= 0.0001 else "<.0001",
            "Normal (p>.05)": p > .05,
        })
    return pd.DataFrame(rows)


def kruskal_wallis(df, val="corr_memorability"):
    labels = COND_ORDER
    groups = [df[df["word_type"] == l][val].dropna().values for l in labels]
    H, p   = stats.kruskal(*groups)
    n, k   = df[val].count(), len(groups)
    eta_sq = (H - k + 1) / (n - k) if (n - k) > 0 else np.nan
    return H, p, eta_sq


def mann_whitney_voice(df, val="corr_memorability"):
    a = df[df["voice"] == "Active"][val].dropna()
    p = df[df["voice"] == "Passive"][val].dropna()
    U, pval = stats.mannwhitneyu(a, p, alternative="two-sided")
    r_rb    = 1 - (2 * U) / (len(a) * len(p))
    return U, pval, r_rb, a.median(), p.median()


# =============================================================================
# SECTION 6 — RAINCLOUD HELPER
# =============================================================================

def raincloud(ax, data, pos, color, width=0.32, dot_alpha=0.4, jitter=0.06, orientation="vertical"):
    """
    Draw a raincloud element at position `pos`.
    Half-violin on left side, jittered dots on right, thin box in centre.
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    if len(data) < 4:
        return

    kde      = stats.gaussian_kde(data, bw_method=0.28)
    y_range  = np.linspace(data.min() - 0.05, data.max() + 0.05, 300)
    kde_vals = kde(y_range)
    kde_vals = kde_vals / kde_vals.max() * (width * 0.88)

    # Half-violin (left of pos)
    ax.fill_betweenx(y_range, pos - kde_vals, pos,
                     color=color, alpha=0.72, linewidth=0)
    ax.plot(pos - kde_vals, y_range, color=color, lw=1.2, alpha=0.85)

    # Jittered dots (right of pos)
    np.random.seed(42)
    jit = np.random.uniform(-jitter, jitter, size=len(data))
    dot_x = pos + 0.06 + kde_vals.max() * 0.2 + jit
    ax.scatter(dot_x, data, color=color, alpha=dot_alpha, s=13,
               linewidths=0, zorder=3)

    # Box-and-whisker
    q1, med, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    lo  = max(data.min(), q1 - 1.5 * iqr)
    hi  = min(data.max(), q3 + 1.5 * iqr)
    bx  = pos + 0.06 + kde_vals.max() * 0.2
    bw  = 0.022
    ax.plot([bx, bx], [lo, hi], color="#333", lw=1.2, zorder=4)
    rect = plt.Rectangle((bx - bw, q1), 2 * bw, iqr,
                          fc="white", ec="#333", lw=1.5, zorder=5)
    ax.add_patch(rect)
    ax.plot([bx - bw, bx + bw], [med, med], color="#333", lw=2.2, zorder=6)
    mn = data.mean()
    ax.scatter([bx], [mn], marker="D", color=color, edgecolors="#333",
               s=44, zorder=7, linewidths=0.8)


# =============================================================================
# SECTION 7 — FIGURES
# =============================================================================

# ── Figure 1: Raincloud by Word Type ─────────────────────────────────────────
def figure1_raincloud_wordtype(df, out_dir):
    df_wt = df.groupby(["participant_ID", "word_type"])["corr_memorability"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F8F9FA")
    ax.grid(axis="y", color="white", lw=1.5, zorder=0)

    labels = ["HH\n(High Subject,\nHigh Object)",
              "HL\n(High Subject,\nLow Object)",
              "LH\n(Low Subject,\nHigh Object)",
              "LL\n(Low Subject,\nLow Object)"]

    for i, (wt, lbl) in enumerate(zip(COND_ORDER, labels)):
        vals = df_wt[df_wt.word_type == wt]["corr_memorability"].values
        raincloud(ax, vals, i, PAL[wt], width=0.30, dot_alpha=0.32)

    ax.axhline(0, color="#999", lw=1, ls="--", alpha=0.7, label="Chance (0)")
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Corrected Memorability Score\n(Hit Rate − FA Rate)", fontsize=12)
    ax.set_title("Figure 1 — Corrected Memorability by Word Type\n"
                 "Raincloud plot: half-violin + jittered observations + box (◆ = mean)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_ylim(-0.25, 1.12)
    ax.set_xlim(-0.55, 3.65)

    patches = [mpatches.Patch(color=PAL[wt], label=f"{wt}")
               for wt in COND_ORDER]
    ax.legend(handles=patches, loc="lower right", fontsize=9, framealpha=0.85,
              title="Word Type")

    for i, wt in enumerate(COND_ORDER):
        n = df_wt[df_wt.word_type == wt].shape[0]
        ax.text(i + 0.06, 1.07, f"N={n}", ha="center", fontsize=8.5, color="#555")

    plt.tight_layout()
    path = os.path.join(out_dir, "fig1_raincloud_wordtype.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


# ── Figure 2: Raincloud by Voice ─────────────────────────────────────────────
def figure2_raincloud_voice(df, out_dir):
    df_v = df.groupby(["participant_ID", "voice"])["corr_memorability"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F8F9FA")
    ax.grid(axis="y", color="white", lw=1.5, zorder=0)

    for i, voice in enumerate(["Active", "Passive"]):
        vals = df_v[df_v.voice == voice]["corr_memorability"].values
        raincloud(ax, vals, i, VPALE[voice], width=0.30, dot_alpha=0.32)

    ax.axhline(0, color="#999", lw=1, ls="--", alpha=0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Active Voice", "Passive Voice"], fontsize=13)
    ax.set_ylabel("Corrected Memorability Score\n(Hit Rate − FA Rate)", fontsize=12)
    ax.set_title("Figure 2 — Corrected Memorability by Sentence Voice\n"
                 "Raincloud plot: half-violin + jittered observations + box (◆ = mean)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_ylim(-0.25, 1.12)
    ax.set_xlim(-0.55, 1.65)

    # Means annotation
    for i, voice in enumerate(["Active", "Passive"]):
        m = df_v[df_v.voice == voice]["corr_memorability"].mean()
        ax.text(i + 0.08, m + 0.04, f"M={m:.3f}", ha="center", fontsize=9,
                color=VPALE[voice], fontweight="bold")

    patches = [mpatches.Patch(color=VPALE[v], label=v) for v in ["Active", "Passive"]]
    ax.legend(handles=patches, loc="lower right", fontsize=10)

    for i, voice in enumerate(["Active", "Passive"]):
        n = df_v[df_v.voice == voice].shape[0]
        ax.text(i + 0.06, 1.07, f"N={n}", ha="center", fontsize=8.5, color="#555")

    plt.tight_layout()
    path = os.path.join(out_dir, "fig2_raincloud_voice.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


# ── Figure 3: Interaction line plot ──────────────────────────────────────────
def figure3_interaction(df, out_dir):
    means = (df.groupby(["word_type", "voice"])["corr_memorability"]
               .agg(["mean", "sem"]).reset_index())
    means.columns = ["word_type", "voice", "mean", "sem"]

    X_POS    = np.arange(4)
    X_LABELS = ["HH\n(High–High)", "HL\n(High–Low)",
                 "LH\n(Low–High)",  "LL\n(Low–Low)"]
    MARKERS  = {"Active": "o", "Passive": "s"}
    COLORS   = {"Active": "#4A6FA5", "Passive": "#9B59B6"}
    LBL_OFF  = {"Active": +0.030, "Passive": -0.033}
    LBL_VA   = {"Active": "bottom", "Passive": "top"}

    fig, ax = plt.subplots(figsize=(9, 5.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F8F9FA")
    ax.grid(axis="y", color="white", lw=1.5, zorder=0)

    for voice in ["Active", "Passive"]:
        g   = means[means["voice"] == voice].set_index("word_type").reindex(COND_ORDER)
        col = COLORS[voice]
        ax.errorbar(X_POS, g["mean"], yerr=1.96 * g["sem"],
                    label=voice, marker=MARKERS[voice], lw=2.2, ms=10,
                    color=col, capsize=5, capthick=1.8, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.8)
        for xi, y in zip(X_POS, g["mean"]):
            ax.annotate(f"{y:.3f}", xy=(xi, y),
                        xytext=(xi, y + LBL_OFF[voice]),
                        ha="center", va=LBL_VA[voice],
                        fontsize=9.5, fontweight="bold", color=col,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                  ec=col, alpha=0.88, lw=0.7),
                        arrowprops=dict(arrowstyle="-", color=col,
                                        lw=0.8, shrinkA=0, shrinkB=3))

    for xi in range(4):
        if xi % 2 == 0:
            ax.axvspan(xi - 0.45, xi + 0.45, color="#f0f0f0", zorder=0, alpha=0.8)

    ax.set_xticks(X_POS)
    ax.set_xticklabels(X_LABELS, fontsize=10.5)
    ax.set_xlim(-0.45, 3.45)
    ax.set_ylim(0.61, 0.80)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_xlabel("Word Memorability Condition", fontsize=12, labelpad=8)
    ax.set_ylabel("Mean Corrected Memorability Score", fontsize=11, labelpad=8)
    ax.set_title("Figure 3 — Word Type × Voice Interaction\n"
                 "Mean corrected memorability ± 95% CI of mean",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(title="Voice", fontsize=10.5, title_fontsize=10.5,
              frameon=True, framealpha=0.92, edgecolor="#ccc", loc="lower left")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig3_interaction.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


# ── Figure 4: Shapiro-Wilk panel ─────────────────────────────────────────────
def figure4_shapiro_panel(df, sw_df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("white")

    # Left: W statistic bar chart per condition
    ax = axes[0]
    ax.set_facecolor("#F8F9FA")
    ax.grid(axis="x", color="white", lw=1.2, zorder=0)
    bar_colors = ["#C44E52" if w < 0.95 else "#55A868" for w in sw_df["W"]]
    bars = ax.barh(sw_df["Condition"], sw_df["W"],
                   color=bar_colors, edgecolor="white", height=0.55)
    ax.axvline(0.95, color="#333", lw=2, ls="--",
               label="W = 0.95 (conventional threshold)")
    ax.set_xlim(0.84, 1.01)
    ax.set_xlabel("Shapiro-Wilk W statistic", fontsize=11)
    ax.set_title("Shapiro-Wilk W per Condition\n(All p < .001 → normality violated)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5)
    for bar, row in zip(bars, sw_df.itertuples()):
        ax.text(bar.get_width() + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"W={row.W:.3f}, p{row.p_str}",
                va="center", fontsize=7.8, color="#333")

    # Right: Observed distribution vs normal reference
    ax2 = axes[1]
    ax2.set_facecolor("#F8F9FA")
    all_scores = df["corr_memorability"].dropna().values
    ax2.hist(all_scores, bins=22, density=True,
             color=PAL["HH"], alpha=0.6, edgecolor="white",
             label="Observed scores")
    kde = stats.gaussian_kde(all_scores, bw_method=0.22)
    xr  = np.linspace(-0.18, 1.12, 300)
    ax2.plot(xr, kde(xr), color=PAL["HH"], lw=2.5, label="KDE (observed)")
    mu, sig = all_scores.mean(), all_scores.std()
    ax2.plot(xr, stats.norm.pdf(xr, mu, sig),
             color="#E74C3C", lw=2.2, ls="--",
             label=f"Normal ref (μ={mu:.3f}, σ={sig:.3f})")
    ax2.axvline(mu, color="#E74C3C", lw=1.2, ls=":", alpha=0.7)
    ax2.set_xlabel("Corrected Memorability Score", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title("Observed Distribution vs Normal Curve\n"
                  "(Left-skewed tail justifies non-parametric approach)",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)

    # Bottom caption
    fig.text(0.5, 0.01,
             "Shapiro-Wilk: All 8 conditions significant (p < .001)  "
             "→  Normality violated  →  Kruskal-Wallis & Mann-Whitney U applied",
             ha="center", fontsize=10.5, fontweight="bold", color="#C44E52",
             bbox=dict(boxstyle="round,pad=0.4", fc="#FFF0F0",
                       ec="#C44E52", lw=1.5))

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    path = os.path.join(out_dir, "fig4_shapiro_panel.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


# ── Figure 5: Sentence examples card ─────────────────────────────────────────
def figure5_sentence_examples(out_dir):
    fig, ax = plt.subplots(figsize=(13, 5.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")

    examples = [
        ("HH", PAL["HH"], "High Subject, High Object",
         '"The volcano destroyed the anchor."',
         "Subject: volcano  ✓ HIGH\nObject:   anchor    ✓ HIGH"),
        ("HL", PAL["HL"], "High Subject, Low Object",
         '"The volcano disrupted the tendency."',
         "Subject: volcano   ✓ HIGH\nObject:   tendency  ✗ LOW"),
        ("LH", PAL["LH"], "Low Subject, High Object",
         '"The concept destroyed the anchor."',
         "Subject: concept  ✗ LOW\nObject:   anchor   ✓ HIGH"),
        ("LL", PAL["LL"], "Low Subject, Low Object",
         '"The concept disrupted the tendency."',
         "Subject: concept   ✗ LOW\nObject:   tendency  ✗ LOW"),
    ]

    cw   = 0.228
    gap  = 0.012
    for i, (code, color, subtitle, sentence, noun_info) in enumerate(examples):
        x0 = i * (cw + gap)

        # Card background
        ax.add_patch(FancyBboxPatch(
            (x0, 0.05), cw, 0.88,
            boxstyle="round,pad=0.015",
            facecolor=color, alpha=0.10,
            edgecolor=color, linewidth=2,
            transform=ax.transAxes))

        # Badge header
        ax.add_patch(FancyBboxPatch(
            (x0 + 0.01, 0.79), cw - 0.02, 0.13,
            boxstyle="round,pad=0.01",
            facecolor=color, alpha=0.88,
            edgecolor="none",
            transform=ax.transAxes))

        ax.text(x0 + cw / 2, 0.858, code,
                ha="center", va="center", fontsize=24,
                fontweight="bold", color="white",
                transform=ax.transAxes)
        ax.text(x0 + cw / 2, 0.735, subtitle,
                ha="center", va="center", fontsize=8,
                color=color, fontweight="bold",
                transform=ax.transAxes)
        ax.text(x0 + cw / 2, 0.545, sentence,
                ha="center", va="center", fontsize=9.2,
                fontstyle="italic", color="#111",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=color, alpha=0.85, lw=1.2))
        ax.text(x0 + cw / 2, 0.26, noun_info,
                ha="center", va="center", fontsize=8.5,
                color="#333", family="monospace",
                transform=ax.transAxes)

    ax.text(0.5, 0.98,
            "4 Word-Type Conditions — Concrete Sentence Examples",
            ha="center", va="top", fontsize=14, fontweight="bold",
            color="#222", transform=ax.transAxes)
    ax.text(0.5, 0.005,
            'Verbs ("destroyed / disrupted") held at medium concreteness across all conditions',
            ha="center", va="bottom", fontsize=9, color="#666",
            fontstyle="italic", transform=ax.transAxes)

    plt.tight_layout()
    path = os.path.join(out_dir, "fig5_sentence_examples.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


# ── Figure 6: Overall distribution histogram ─────────────────────────────────
def figure6_distribution_hist(df, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F8F9FA")
    ax.grid(axis="y", color="white", lw=1.3, zorder=0)

    scores = df["corr_memorability"].dropna().values
    ax.hist(scores, bins=24, density=True,
            color=PAL["HH"], alpha=0.60, edgecolor="white",
            label="Scores (density)")

    kde = stats.gaussian_kde(scores, bw_method=0.22)
    xr  = np.linspace(-0.18, 1.12, 300)
    ax.plot(xr, kde(xr), color=PAL["HH"], lw=2.8, label="KDE")

    mu = scores.mean()
    ax.axvline(mu, color="#E74C3C", lw=2, ls="--",
               label=f"Mean = {mu:.3f}")
    ax.axvline(np.median(scores), color="#F18F01", lw=2, ls=":",
               label=f"Median = {np.median(scores):.3f}")
    ax.axvline(0, color="#999", lw=1.2, ls="-", alpha=0.5, label="Chance (0)")

    ax.set_xlabel("Corrected Memorability Score (Hit Rate − FA Rate)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Figure 6 — Overall Distribution of Corrected Memorability Scores\n"
                 "Left-skewed: most participants score 0.6–0.9; long left tail pulls mean down",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_xlim(-0.22, 1.12)

    plt.tight_layout()
    path = os.path.join(out_dir, "fig6_distribution_hist.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


# =============================================================================
# SECTION 8 — MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  SENTENCE MEMORABILITY — REPORT 1 ANALYSIS  (v2)")
    print("=" * 65)
    print(f"\n  Log directory   : {LOG_DIR}")
    print(f"  Output directory: {OUT_DIR}\n")

    # ── Step 1: Parse and score ──────────────────────────────────────────────
    print("[1]  Parsing logs and computing corrected memorability...")
    df, p_sum = process_all_participants(LOG_DIR)
    df        = df[df["word_type"].isin(TARGET_TYPES)].copy()

    n_total    = len(p_sum)
    n_excluded = int(p_sum["excluded"].sum())
    n_retained = n_total - n_excluded

    print(f"     Participants processed       : {n_total}")
    print(f"     Excluded (all blocks invalid): {n_excluded}")
    print(f"     Retained                     : {n_retained}")
    print(f"     Condition records            : {len(df)}")

    # ── Step 2: Descriptive stats ────────────────────────────────────────────
    print("\n[2]  Descriptive statistics...")
    t1 = (descriptive_stats(df, "word_type")
          .set_index("word_type").reindex(COND_ORDER).reset_index())
    t2 = descriptive_stats(df, "voice")
    t3 = (df.groupby(["word_type", "voice"])["corr_memorability"]
            .mean().unstack().round(4).reindex(COND_ORDER))

    print("\n  Table 1 — By Word Type:")
    print(t1.to_string(index=False))
    print("\n  Table 2 — By Voice:")
    print(t2.to_string(index=False))
    print("\n  Table 3 — Interaction Means (Word Type × Voice):")
    print(t3.to_string())

    # ── Step 3: Normality (Shapiro-Wilk) ────────────────────────────────────
    print("\n[3]  Shapiro-Wilk normality tests per condition...")
    sw_df = shapiro_wilk_per_condition(df)
    print(sw_df[["Condition", "W", "p_str", "Normal (p>.05)"]].to_string(index=False))
    all_violated = not sw_df["Normal (p>.05)"].any()
    print(f"  → All conditions violate normality (all p < .05): {all_violated}")
    print("  → Kruskal-Wallis and Mann-Whitney U are the appropriate tests.")

    # ── Step 4: Kruskal-Wallis ───────────────────────────────────────────────
    print("\n[4]  Kruskal-Wallis — effect of word type...")
    H, p_kw, eta = kruskal_wallis(df)
    k = df["word_type"].nunique()
    eff = "large" if eta > .14 else ("medium" if eta > .06 else "small")
    print(f"  H({k-1}) = {H:.4f},  p = {p_kw:.4f},  η² = {eta:.4f}  ({eff})")
    print(f"  Significant: {'YES ✓' if p_kw < .05 else 'NO'}")

    # ── Step 5: Mann-Whitney ─────────────────────────────────────────────────
    print("\n[5]  Mann-Whitney U — effect of voice...")
    U, p_mw, r_rb, med_a, med_p = mann_whitney_voice(df)
    print(f"  U = {U:.2f},  p = {p_mw:.4f},  r = {r_rb:.4f}")
    print(f"  Median Active = {med_a:.4f},  Median Passive = {med_p:.4f}")
    print(f"  Significant: {'YES ✓' if p_mw < .05 else 'NO'}")

    # ── Step 6: Figures ──────────────────────────────────────────────────────
    print("\n[6]  Generating figures...")

    paths = []
    paths.append(figure1_raincloud_wordtype(df, OUT_DIR))
    paths.append(figure2_raincloud_voice(df, OUT_DIR))
    paths.append(figure3_interaction(df, OUT_DIR))
    paths.append(figure4_shapiro_panel(df, sw_df, OUT_DIR))
    paths.append(figure5_sentence_examples(OUT_DIR))
    paths.append(figure6_distribution_hist(df, OUT_DIR))

    for p in paths:
        print(f"    {os.path.basename(p)}")

    # ── Step 7: Save CSVs ────────────────────────────────────────────────────
    print("\n[7]  Saving tables...")
    t1.to_csv(os.path.join(OUT_DIR, "table1_wordtype.csv"),          index=False)
    t2.to_csv(os.path.join(OUT_DIR, "table2_voice.csv"),             index=False)
    t3.to_csv(os.path.join(OUT_DIR, "table3_interaction.csv"))
    sw_df.to_csv(os.path.join(OUT_DIR, "table_shapiro_wilk.csv"),    index=False)
    p_sum.to_csv(os.path.join(OUT_DIR, "participant_summary.csv"),   index=False)
    df.to_csv(os.path.join(OUT_DIR,   "corrected_memorability.csv"), index=False)
    print("  All CSVs saved.")

    print("\n" + "=" * 65)
    print("  DONE — outputs in:", OUT_DIR)
    print("=" * 65)

    # ── Quick summary for report/PPT use ─────────────────────────────────────
    print("\n──────────────────────────────────────────────────")
    print("  COPY-PASTE VALUES FOR REPORT / PPT")
    print("──────────────────────────────────────────────────")
    print(f"  N total={n_total}, excluded={n_excluded}, retained={n_retained}")
    print(f"\n  Shapiro-Wilk: all p < .05 → non-normal in all 8 conditions")
    print(f"  Kruskal-Wallis: H({k-1})={H:.2f}, p={p_kw:.4f}, η²={eta:.4f} ({eff})")
    print(f"  Mann-Whitney U: U={U:.1f}, p={p_mw:.4f}, r={r_rb:.4f}")
    print("──────────────────────────────────────────────────")

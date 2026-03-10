"""
=============================================================================
 Sentence Memorability Experiment — Updated Visualizations
 Team: Vishnu Varun | Pavan Karke | Susheel Krishna
 -----------------------------------------------------------------------------
 This script takes the processed data from `results2/corrected_memorability.csv`
 and generates clean, non-cognitive-heavy plots as recommended in the review.
 
 OUTPUTS:
   ├── fig1_cleaned_wordtype.png     — Bar chart: memorability × word type (with 95% CI)
   ├── fig2_cleaned_voice.png        — Bar chart: memorability × voice (with 95% CI)
   ├── fig3_cleaned_interaction.png  — Clean interaction line plot (word type × voice)
   ├── fig4_violin_wordtype.png      — Violin plot: memorability × word type
   ├── fig5_violin_voice.png         — Violin plot: memorability × voice
=============================================================================
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# ── CONFIGURATION
# =============================================================================
_HERE   = os.path.dirname(os.path.abspath(__file__))
# Note: Ensure that results2/corrected_memorability.csv exists from previous runs!
IN_FILE = os.path.join(_HERE, "results2", "corrected_memorability.csv")
OUT_DIR = os.path.join(_HERE, "results2")
os.makedirs(OUT_DIR, exist_ok=True)

# Colour palettes ensuring consistency with previous charts
PAL        = {"HH": "#4C72B0", "HL": "#DD8452", "LH": "#55A868", "LL": "#C44E52"}
VPALE      = {"Active": "#4A6FA5", "Passive": "#9B59B6"}
COND_ORDER = ["HH", "HL", "LH", "LL"]

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "font.size"        : 11,
})

# =============================================================================
# ── FIGURE 1: Bar Chart by Word Type (95% CI)
# =============================================================================
def plot_wordtype_bar(df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # seaborn barplot automatically computes means and 95% CI error bars
    sns.barplot(
        data=df, 
        x="word_type", 
        y="corr_memorability", 
        order=COND_ORDER,
        palette=PAL, 
        capsize=0.1, 
        errorbar=("ci", 95),
        err_kws={'linewidth': 1.5, 'color': 'black'},
        ax=ax
    )

    ax.axhline(0, color="#999", lw=1, ls="--", alpha=0.7, label="Chance (0)")
    
    labels = ["HH\n(High Subject,\nHigh Object)",
              "HL\n(High Subject,\nLow Object)",
              "LH\n(Low Subject,\nHigh Object)",
              "LL\n(Low Subject,\nLow Object)"]
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=11)
    
    ax.set_ylabel("Mean Corrected Memorability Score", fontsize=12, labelpad=10)
    ax.set_xlabel("Word Memorability Condition", fontsize=12, labelpad=10)
    ax.set_title("Corrected Memorability by Word Type\n(Bars represent Means ± 95% CI)",
                 fontsize=13, fontweight="bold", pad=15)
    
    plt.tight_layout()
    path = os.path.join(out_dir, "fig1_cleaned_wordtype.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    return path

# =============================================================================
# ── FIGURE 2: Bar Chart by Voice (95% CI)
# =============================================================================
def plot_voice_bar(df, out_dir):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    sns.barplot(
        data=df, 
        x="voice", 
        y="corr_memorability", 
        order=["Active", "Passive"],
        palette=VPALE, 
        capsize=0.1, 
        err_kws={'linewidth': 1.5, 'color': 'black'},
        width=0.6,
        ax=ax
    )

    ax.axhline(0, color="#999", lw=1, ls="--", alpha=0.7)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Active Voice", "Passive Voice"], fontsize=12)
    
    ax.set_ylabel("Mean Corrected Memorability Score", fontsize=12, labelpad=10)
    ax.set_xlabel("Sentence Grammatical Voice", fontsize=12, labelpad=10)
    ax.set_title("Corrected Memorability by Voice\n(Bars represent Means ± 95% CI)",
                 fontsize=13, fontweight="bold", pad=15)
                 
    # Add simple mean text inside the bars for quick reference
    means = df.groupby("voice")["corr_memorability"].mean()
    for i, voice in enumerate(["Active", "Passive"]):
        m = means[voice]
        ax.text(i, m / 2, f"M = {m:.3f}", ha="center", va="center", 
                fontsize=11, fontweight="bold", color="white")
    
    plt.tight_layout()
    path = os.path.join(out_dir, "fig2_cleaned_voice.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    return path

# =============================================================================
# ── FIGURE 3: Cleaned Interaction Plot
# =============================================================================
def plot_interaction_clean(df, out_dir):
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # seaborn pointplot creates beautiful interaction lines with CI
    sns.pointplot(
        data=df, 
        x="word_type", 
        y="corr_memorability", 
        hue="voice",
        order=COND_ORDER,
        hue_order=["Active", "Passive"],
        palette=VPALE,
        markers=["o", "s"], 
        linestyles=["-", "--"],
        capsize=0.05,
        dodge=True, # slightly offsets the lines so error bars don't overlap completely
        ax=ax
    )
    
    X_LABELS = ["HH\n(High–High)", "HL\n(High–Low)",
                "LH\n(Low–High)",  "LL\n(Low–Low)"]
    ax.set_xticklabels(X_LABELS, fontsize=11)
    
    ax.set_ylabel("Mean Corrected Memorability Score", fontsize=12, labelpad=10)
    ax.set_xlabel("Word Memorability Condition", fontsize=12, labelpad=10)
    ax.set_title("Word Type × Voice Interaction\n(Means ± 95% Confidence Intervals)",
                 fontsize=13, fontweight="bold", pad=15)
                 
    ax.legend(title="Voice", fontsize=11, title_fontsize=12, frameon=True, edgecolor="#ccc")
    
    plt.tight_layout()
    path = os.path.join(out_dir, "fig3_cleaned_interaction.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    return path

# =============================================================================
# ── FIGURE 4: Violin Plot by Word Type
# =============================================================================
def plot_wordtype_violin(df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.violinplot(
        data=df, 
        x="word_type", 
        y="corr_memorability", 
        hue="word_type",
        order=COND_ORDER,
        palette=PAL, 
        inner="box",
        legend=False,
        ax=ax
    )

    ax.axhline(0, color="#999", lw=1, ls="--", alpha=0.7, label="Chance (0)")
    
    labels = ["HH\n(High Subject,\nHigh Object)",
              "HL\n(High Subject,\nLow Object)",
              "LH\n(Low Subject,\nHigh Object)",
              "LL\n(Low Subject,\nLow Object)"]
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=11)
    
    ax.set_ylabel("Corrected Memorability Score", fontsize=12, labelpad=10)
    ax.set_xlabel("Word Memorability Condition", fontsize=12, labelpad=10)
    ax.set_title("Corrected Memorability by Word Type (Violin Plot)\n(Width represents density; inner box shows IQR & median)",
                 fontsize=13, fontweight="bold", pad=15)
    
    plt.tight_layout()
    path = os.path.join(out_dir, "fig4_violin_wordtype.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    return path

# =============================================================================
# ── FIGURE 5: Violin Plot by Voice
# =============================================================================
def plot_voice_violin(df, out_dir):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    sns.violinplot(
        data=df, 
        x="voice", 
        y="corr_memorability", 
        hue="voice",
        order=["Active", "Passive"],
        palette=VPALE, 
        inner="box",
        width=0.6,
        legend=False,
        ax=ax
    )

    ax.axhline(0, color="#999", lw=1, ls="--", alpha=0.7)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Active Voice", "Passive Voice"], fontsize=12)
    
    ax.set_ylabel("Corrected Memorability Score", fontsize=12, labelpad=10)
    ax.set_xlabel("Sentence Grammatical Voice", fontsize=12, labelpad=10)
    ax.set_title("Corrected Memorability by Voice (Violin Plot)\n(Width represents density; inner box shows IQR & median)",
                 fontsize=13, fontweight="bold", pad=15)
                 
    plt.tight_layout()
    path = os.path.join(out_dir, "fig5_violin_voice.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    return path

# =============================================================================
# ── MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  GENERATING CLEANED PLOTS")
    print("=" * 65)
    
    if not os.path.exists(IN_FILE):
        print(f"[ERROR] Could not find {IN_FILE}")
        print("Please ensure you have run the main processing script first to generate the CSV.")
    else:
        print(f"Loading data from {IN_FILE}...")
        df = pd.read_csv(IN_FILE)
        
        print("Generating Figure 1: Cleaned Word Type Bar Chart...")
        p1 = plot_wordtype_bar(df, OUT_DIR)
        
        print("Generating Figure 2: Cleaned Voice Bar Chart...")
        p2 = plot_voice_bar(df, OUT_DIR)
        
        print("Generating Figure 3: Cleaned Interaction Plot...")
        p3 = plot_interaction_clean(df, OUT_DIR)
        
        print("Generating Figure 4: Violin Plot by Word Type...")
        p4 = plot_wordtype_violin(df, OUT_DIR)
        
        print("Generating Figure 5: Violin Plot by Voice...")
        p5 = plot_voice_violin(df, OUT_DIR)
        
        print("\nSUCCESS! Saved cleaned plots to:")
        print(f"  - {p1}")
        print(f"  - {p2}")
        print(f"  - {p3}")
        print(f"  - {p4}")
        print(f"  - {p5}")
        print("=" * 65)

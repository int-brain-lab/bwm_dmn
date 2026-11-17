#!/usr/bin/env python3
"""Reproduce SI Lorenzo comparison from saved result files.

Call ``make_si_lorenzo()`` to create PNG, SVG and caption. This plotting step
does not refit models; upstream result tables are made by
``recompute_panel_f_rank5_gpu.py`` and ``recompute_panels_de.py``.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm, pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
LORENZO = HERE.parent
DMN = HERE.parents[1]
RED, REDLINE = "#d95f5f", "#c94747"
FONT_SCALE = 1.30


def load_top_results(results_dir=HERE):
    """Return rank-5 RRR, PETH and rank-3 RRR regional silhouette tables."""
    area_file = LORENZO / "clustering-analysis/data/area_list.csv"
    areas = pd.read_csv(area_file, header=None)[0].astype(str).tolist()
    positions = {area: i for i, area in enumerate(areas)}
    results_dir = Path(results_dir)
    rrr_file = results_dir / "panel_d_rrr_silhouette.csv"
    if not rrr_file.exists():
        rrr_file = LORENZO / "sillouette_score/reproduction_audit/maximal_harris_scored_cortical_points.csv"
    raw_rrr = pd.read_csv(rrr_file)
    rrr = pd.DataFrame({"region": raw_rrr.region, "y": raw_rrr.silhouette_z})
    rrr["x"] = rrr.region.map(positions)
    peth_file = results_dir / "panel_e_peth_silhouette.csv"
    if not peth_file.exists():
        peth_file = LORENZO / "sillouette_score/dmn_region_results.csv"
    raw_peth = pd.read_csv(peth_file)
    peth = pd.DataFrame({"region": raw_peth.region, "y": raw_peth.silhouette_z})
    peth["x"] = peth.region.map(positions)
    rank3_file = results_dir / "panel_f_rank3_silhouette.csv"
    if not rank3_file.exists():
        rank3_file = LORENZO / "sillouette_score/local_rrr_region_results.csv"
    raw_rank3 = pd.read_csv(rank3_file)
    rank3 = pd.DataFrame({"region": raw_rank3.region, "y": raw_rank3.silhouette_z})
    rank3["x"] = rank3.region.map(positions)
    for name, data in zip("def", (rrr, peth, rank3)):
        if data.x.isna().any():
            missing = data.loc[data.x.isna(), "region"].tolist()
            raise ValueError(f"Panel {name}: regions absent from hierarchy: {missing}")
    return rrr, peth, rank3


# RRR variable (Posani Fig. 2c) matched to the BWM decoder of the same quantity.
DECODING_PAIRS = [("side", "Side", "stimside", "Stimulus side"),
                  ("choice", "Choice", "choice", "Choice"),
                  ("outcome", "Outcome", "feedback", "Feedback"),
                  ("wheel", "Wheel", "wheel-speed", "Wheel speed"),
                  ("wheel", "Wheel", "wheel-velocity", "Wheel velocity")]


def draw_decoding_comparison(axes, csv_file):
    """Draw one across-region scatter per matched RRR/decoding variable."""
    data = pd.read_csv(csv_file)
    stats = []
    for ax, (avar, alabel, dvar, dlabel) in zip(axes, DECODING_PAIRS):
        x, y = data[f"alpha_{avar}_z"], data[f"dec_{dvar}_z"]
        good = np.isfinite(x) & np.isfinite(y)
        x, y = x[good], y[good]
        rho, p = spearmanr(x, y)
        stats.append((alabel, dlabel, rho, p, int(good.sum())))
        sns.regplot(x=x, y=y, ax=ax, scatter=False, ci=95, n_boot=5000, seed=0,
                    color=REDLINE, line_kws={"lw": 2})
        ax.scatter(x, y, s=24, c=RED, edgecolor="black", linewidth=.65, zorder=3)
        ax.text(.03, .97, f"ρ = {rho:.2f}\nP = {p:.2g}", transform=ax.transAxes,
                ha="left", va="top", fontsize=9 * FONT_SCALE,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.8, pad=1.5))
        ax.set(title=f"{alabel} vs {dlabel.lower()}" if alabel != dlabel else alabel,
               xlabel="RRR selectivity (z)", ylabel="")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("BWM decoding (z)")
    return stats


def draw_top(ax, data, title, ylabel, legend=False, region_fontsize=8.4 * FONT_SCALE):
    """Draw one region-level silhouette panel and return its statistics."""
    rho, p = spearmanr(data.x, data.y)
    threshold = norm.ppf(1 - .05 / len(data))
    sns.regplot(data=data, x="x", y="y", ax=ax, scatter=False, ci=95,
                n_boot=5000, seed=0, color=REDLINE, line_kws={"lw": 2})
    ax.scatter(data.x, data.y, s=24, c=RED, edgecolor="black", linewidth=.65, zorder=3)
    for row in data.itertuples():
        ax.annotate(row.region, (row.x, row.y), xytext=(3, 3),
                    textcoords="offset points", fontsize=region_fontsize)
    ax.axhline(threshold, color=".48", ls="--", lw=1.5,
               label=f"Bonferroni clustering threshold (z={threshold:.2f})")
    ax.text(.97, .97, f"ρ = {rho:.2f}\nP = {p:.3g}\nn = {len(data)} regions",
            transform=ax.transAxes, ha="right", va="top", fontsize=9 * FONT_SCALE,
            bbox=dict(facecolor="white", edgecolor="none", alpha=.8, pad=1.5))
    if legend:
        ax.legend(frameon=False, fontsize=7 * FONT_SCALE, loc="lower left")
    ax.set(title=title, xlabel="Position in the hierarchy", ylabel=ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    return rho, p


def draw_bottom(ax, csv_file, xcol, ycol, title, xlabel, ylabel, statistic,
                qap_text="< 10⁻⁴"):
    """Draw one saved region-pair table and return its correlation."""
    data = pd.read_csv(csv_file)
    x, y = data[xcol].to_numpy(), data[ycol].to_numpy()
    value = spearmanr(x, y).statistic if statistic == "Spearman" else pearsonr(x, y).statistic
    ax.scatter(x, y, s=4, c="black", alpha=.32, edgecolors="none", rasterized=True)
    coef = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 200)
    ax.plot(xx, np.polyval(coef, xx), color=REDLINE, lw=1.8)
    symbol = "ρ" if statistic == "Spearman" else "r"
    ax.text(.97, .97, f"{symbol} = {value:.2f}\nQAP P {qap_text}\nn = {len(data)} region pairs",
            transform=ax.transAxes, ha="right", va="top", fontsize=9 * FONT_SCALE,
            bbox=dict(facecolor="white", edgecolor="none", alpha=.8, pad=1.5))
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    return value


def make_si_lorenzo(output_dir=DMN / "figs", results_dir=HERE,
                    basename="si_lorenzo", dpi=300):
    """Build all seven panels from results and return generated file paths."""
    output_dir, results_dir = Path(output_dir), Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9 * FONT_SCALE,
                         "axes.linewidth": .8, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(18, 15.2), facecolor="white")
    gs = fig.add_gridspec(3, 3, hspace=.36, wspace=.25, height_ratios=[1, 1, .78],
                          left=.055, right=.98, top=.97, bottom=.055)
    top = [fig.add_subplot(gs[0, i]) for i in range(3)]
    bottom = [fig.add_subplot(gs[1, i]) for i in range(3)]
    for ax in bottom[1:]:
        ax.sharex(bottom[0]); ax.sharey(bottom[0])
    dec_gs = gs[2, :].subgridspec(1, len(DECODING_PAIRS), wspace=.3)
    dec_axes = [fig.add_subplot(dec_gs[0, i]) for i in range(len(DECODING_PAIRS))]
    for ax in dec_axes[1:]:
        ax.sharex(dec_axes[0]); ax.sharey(dec_axes[0])
    axes = top + bottom
    rrr, peth, rank3 = load_top_results(results_dir)
    draw_bottom(top[0], results_dir / "panel_f_rank5_pairs.csv",
                "anatomical_connectivity", "selectivity_similarity",
                "RRR similarity versus anatomy",
                "Anatomical connectivity score", "Selectivity similarity (RRR)", "Spearman")
    # This is the identical Posani connectivity matrix/transformation used in a.
    draw_bottom(top[1], results_dir / "panel_d_pairs.csv",
                "anatomical_connectivity", "peth_cosine_similarity",
                "PETH similarity versus anatomy",
                "Anatomical connectivity score",
                "Selectivity similarity (PETH)", "Pearson", "= 0.0030")
    draw_bottom(top[2], results_dir / "panel_e_pairs.csv",
                "dmn_cosine_similarity", "rank5_rrr_cosine_similarity",
                "RRR versus PETH selectivity similarity",
                "Selectivity similarity (PETH)", "Selectivity similarity (RRR)", "Pearson")
    draw_top(bottom[0], rrr, "Rank-5 RRR selectivity clustering",
             "Silhouette score (RRR)", True)
    draw_top(bottom[1], rank3, "Rank-3 (CV-selected) RRR selectivity clustering",
             "Silhouette score (RRR)")
    draw_top(bottom[2], peth, "PETH selectivity clustering",
             "Silhouette score (PETH)")
    dec_stats = draw_decoding_comparison(dec_axes, results_dir / "panel_g_rrr_vs_bwm_decoding.csv")
    bottom[0].set_xlim(0, 42)
    bottom[0].set_ylim(-1, 13)
    dec_axes[0].set_xlim(-3, 3)
    dec_axes[0].set_ylim(-3, 3)
    for i, ax in enumerate(axes):
        locator = MaxNLocator(nbins=2, min_n_ticks=3) if i < 3 else LinearLocator(3)
        ax.xaxis.set_major_locator(locator)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=2, min_n_ticks=3) if i < 3 else LinearLocator(3))
        ax.text(-.12, 1.07, chr(97 + i), transform=ax.transAxes, fontsize=20 * FONT_SCALE,
                fontweight="bold", va="top")
    for ax in dec_axes:
        ax.xaxis.set_major_locator(LinearLocator(3))
        ax.yaxis.set_major_locator(LinearLocator(3))
    dec_axes[0].text(-.3, 1.12, "g", transform=dec_axes[0].transAxes,
                     fontsize=20 * FONT_SCALE, fontweight="bold", va="top")

    survivors = [dlabel.lower() for _, dlabel, _, p, _ in dec_stats if p < .05 / len(dec_stats)]
    bonf_survivors = " and ".join(survivors) if survivors else "none"
    captions = [
        ("a", "RRR selectivity similarity uses each neuron's time-integrated absolute coefficient for every task variable as features; neurons are averaged within each region, features are standardized across regions, and cosine similarity is calculated between regional vectors."),
        ("b", "PETH selectivity similarity uses every PETH time bin as features; neurons are averaged within each region and cosine similarity is calculated between regional vectors. The x-axis uses the same Posani-style anatomical connectivity score as a."),
        ("c", "Restricting to 67 regions with at least 50 UUID-matched neurons (10,477 included), regional DMN PETH geometry covaries with rank-5 RRR time-summed selectivity geometry."),
        ("d", f"Rank-5 RRR silhouette z-score versus Posani hierarchy position for {len(rrr)} regions passing RRR selectivity, neuron-count, and session-dominance criteria."),
        ("e", f"Same analysis as d, but using the rank-3 (λ = 25) RRR model selected automatically by three-fold cross-validation (mean CV R² 0.068 versus 0.066 for rank 5); neuron inclusion uses the rank-3 ΔR² > 0.015, leaving {len(rank3)} regions."),
        ("f", f"PETH silhouette z-score versus the same hierarchy position for {len(peth)} regions passing the PETH neuron-count and session-dominance criteria."),
        ("g", f"Across-region comparison of RRR mean absolute selectivity (Posani et al., Fig. 2c; authors' released coefficients) with BWM regional decoding scores, restricted to the task variables present in both analyses; RRR wheel selectivity is compared with both wheel-speed and wheel-velocity decoding. Each point is one of the {dec_stats[0][4]} cortical regions with ≥30 RRR neurons and ≥3 regional decoding scores; both quantities are z-scored across these regions. Spearman P values are uncorrected; after Bonferroni correction for five comparisons, {bonf_survivors} remain significant (uncorrected P < 0.01)."),
    ]
    note = ("Dashed lines show the one-sided Bonferroni-corrected clustering criterion; "
            "its legend appears only in d. RRR has fewer regions than PETH because RRR first "
            "requires neuron-level prediction improvement over the null (ΔR² > 0.015, evaluated for each rank); "
            "PETH uses all available cortical DMN neurons before both pipelines apply the ≥50-neuron "
            "and session-dominance criteria. Region-pair P values use two-sided 10,000-permutation "
            "region-label QAP tests because pairs sharing regions are not independent.")
    png, svg = output_dir / f"{basename}_panels.png", output_dir / f"{basename}_panels.svg"
    pdf = output_dir / f"{basename}_panels.pdf"
    caption_file = output_dir / f"{basename}_caption.txt"
    fig.savefig(png, dpi=dpi, facecolor="white")
    fig.savefig(svg, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    caption_file.write_text("Figure caption\n" + "\n".join(
        f"**{label}** {description}" for label, description in captions) + "\n" + note + "\n")
    return {"png": png, "svg": svg, "pdf": pdf, "caption": caption_file}


if __name__ == "__main__":
    for kind, path in make_si_lorenzo().items():
        print(f"{kind}: {path}")

#!/usr/bin/env python3
"""Tests for the latency-tiling text (2026-10-13), written to sparse_cortex_test.txt.

1  Cortical enrichment of sparse tiling: 2 x 2 table (cortical vs subcortical neurons x
   sparse tiling vs not), unadjusted (chi-square) and stratified by firing rate
   (Cochran-Mantel-Haenszel over the log-rate bins of cortex_rate_analysis.RATE_BINS:
   common odds ratio, test of OR = 1). Robustness: logistic regression sparse ~ cortical
   + log rate with standard errors clustered by insertion. Neurons and the cortical /
   subcortical split as Fig. 4i,j (../../fig_si/cortex_rate_analysis.load).
2  k-means clusters (Fig. 2, printed 1-based) of dense- and sparse-tiling neurons:
   share of each group vs share of all neurons.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2_contingency
from statsmodels.stats.contingency_tables import StratifiedTable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "fig_si"))
import cortex_rate_analysis as CR  # noqa: E402


def main():
    _, df, _ = CR.load()
    d = df[df.group.notna()].copy()
    y = (d.group == "sparse").to_numpy()
    c = d.ctx.to_numpy()
    lines = [f"neurons with Fig. 4 groups and cortical/subcortical label: {len(d):,} "
             f"(cortical {c.sum():,}, subcortical {(~c).sum():,}; sparse tiling {y.sum():,})"]
    t = np.array([[np.sum(c & y), np.sum(c & ~y)], [np.sum(~c & y), np.sum(~c & ~y)]])
    chi2, p, _, _ = chi2_contingency(t)
    orr = t[0, 0] * t[1, 1] / (t[0, 1] * t[1, 0])
    lines.append(f"unadjusted: sparse fraction cortical {t[0,0]/t[0].sum():.3f}, subcortical "
                 f"{t[1,0]/t[1].sum():.3f} (ratio {t[0,0]/t[0].sum()/(t[1,0]/t[1].sum()):.2f}); "
                 f"OR {orr:.2f}, chi2 {chi2:.0f}, p {p:.1e}")
    b = np.digitize(d.lograte.to_numpy(), CR.RATE_BINS)
    tabs = []
    for k in np.unique(b):
        m = b == k
        tk = np.array([[np.sum(m & c & y), np.sum(m & c & ~y)], [np.sum(m & ~c & y), np.sum(m & ~c & ~y)]])
        if (tk.sum(1) > 0).all():
            tabs.append(tk)
    st = StratifiedTable(tabs)
    res = st.test_null_odds()
    ci = st.oddsratio_pooled_confint()
    lines.append(f"rate-stratified (CMH, {len(tabs)} log-rate bins): common OR {st.oddsratio_pooled:.2f} "
                 f"[95% CI {ci[0]:.2f}-{ci[1]:.2f}], chi2 {res.statistic:.0f}, p {res.pvalue:.1e}; "
                 f"homogeneity of OR across bins p {st.test_equal_odds().pvalue:.2g}")
    X = sm.add_constant(np.column_stack([c.astype(float), d.lograte.to_numpy()]))
    fit = sm.Logit(y.astype(float), X).fit(disp=0, cov_type="cluster",
                                           cov_kwds={"groups": pd.factorize(d.pid)[0]})
    lines.append(f"logistic, insertion-clustered SE ({d.pid.nunique()} insertions): cortical OR "
                 f"{np.exp(fit.params[1]):.2f} [95% CI {np.exp(fit.conf_int()[1][0]):.2f}-"
                 f"{np.exp(fit.conf_int()[1][1]):.2f}], p {fit.pvalues[1]:.1e}; per log10 rate OR "
                 f"{np.exp(fit.params[2]):.2f}")
    fit0 = sm.Logit(y.astype(float), sm.add_constant(c.astype(float))).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(d.pid)[0]})
    lines.append(f"logistic without rate, clustered: cortical OR {np.exp(fit0.params[1]):.2f}, p {fit0.pvalues[1]:.1e}")

    lines.append("")
    allshare = d.kmeans.value_counts(normalize=True)
    for g in ("dense", "sparse"):
        s = d[d.group == g].kmeans.value_counts(normalize=True)
        top = (s / allshare.reindex(s.index)).sort_values(ascending=False)
        lines.append(f"{g} tiling (n={np.sum(d.group == g):,}), k-means clusters (1-based) by share: " + ", ".join(
            f"{k+1}: {100*s[k]:.0f}% vs {100*allshare[k]:.0f}% ({s[k]/allshare[k]:.1f}x)"
            for k in s.sort_values(ascending=False).index[:5]))
        lines.append(f"  most enriched: " + ", ".join(f"{k+1} ({v:.1f}x, {100*s[k]:.0f}%)" for k, v in top.iloc[:5].items()))
    (HERE / "sparse_cortex_test.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

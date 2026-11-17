#!/usr/bin/env python3
"""Recompute every SI-Lorenzo panel from local data, then render the figure.

Typical full rerun (uses the existing local rank-5 fit):
    python reproduce_all.py

True model-level restart (slow; refits RRR and reevaluates three CV folds):
    python reproduce_all.py --refit-rrr
"""
from pathlib import Path
import argparse, json, subprocess, sys, time
import pandas as pd

HERE = Path(__file__).resolve().parent
LORENZO = HERE.parent
SIL = LORENZO / "sillouette_score"
AUDIT = SIL / "reproduction_audit"


def run(script, *args):
    command = [sys.executable, str(script), *map(str, args)]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refit-rrr", action="store_true",
                        help="refit the full rank-5 model and recompute held-out R²")
    parser.add_argument("--n-null", type=int, default=100)
    parser.add_argument("--n-init", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=HERE.parents[1] / "figs")
    args = parser.parse_args()
    started = time.time()

    if args.refit_rrr:
        run(AUDIT / "train_rank5_full.py", "--force")
        run(AUDIT / "evaluate_rank5_cv.py")

    # Region-level silhouette analyses, both using the released clustering rules.
    run(AUDIT / "confirm_rank5.py", "--device", "cuda", "--region-set", "all",
        "--n-null", args.n_null, "--n-init", args.n_init)
    run(SIL / "run_analysis.py", "--analysis", "local_rrr", "--device", "cuda",
        "--n-null", args.n_null, "--n-init", args.n_init)  # CV-selected rank 3
    run(SIL / "run_analysis.py", "--analysis", "dmn", "--device", "cuda",
        "--n-null", args.n_null, "--n-init", args.n_init)

    # Canonical plot inputs; this prevents the compositor from depending on old figures.
    pd.read_csv(AUDIT / "rank5_all_cuda_points.csv").to_csv(
        HERE / "panel_d_rrr_silhouette.csv", index=False)
    pd.read_csv(SIL / "dmn_region_results.csv").to_csv(
        HERE / "panel_e_peth_silhouette.csv", index=False)
    pd.read_csv(SIL / "local_rrr_region_results.csv").to_csv(
        HERE / "panel_f_rank3_silhouette.csv", index=False)
    # Panel g: Posani RRR selectivity vs BWM decoding, from the side-by-side script.
    pd.read_csv(HERE.parents[1] / "alpha_diversity_comparison" /
                "rrr_selectivity_vs_bwm_decoding_values.csv").to_csv(
        HERE / "panel_g_rrr_vs_bwm_decoding.csv", index=False)

    # Region-pair panels a-c. All feature aggregation uses CUDA where applicable.
    run(HERE / "recompute_panel_f_rank5_gpu.py")
    run(HERE / "recompute_panel_e_posani_connectivity_gpu.py")
    run(HERE / "recompute_panel_f_matched_rank5_gpu.py")

    from compose_si_lorenzo_native import make_si_lorenzo
    outputs = make_si_lorenzo(output_dir=args.output_dir, results_dir=HERE)
    if args.output_dir.resolve() == (HERE.parents[1] / "figs").resolve():
        run(HERE / "build_si_lorenzo_latex.py")
        outputs["latex"] = args.output_dir / "si_lorenzo.tex"
        outputs["compiled_pdf"] = args.output_dir / "si_lorenzo.pdf"
    manifest = {
        "elapsed_seconds": time.time() - started,
        "device": "cuda",
        "refit_rrr": args.refit_rrr,
        "n_null": args.n_null,
        "n_init": args.n_init,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    (HERE / "reproduction_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

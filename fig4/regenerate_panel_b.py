#!/usr/bin/env python3
"""Regenerate Fig. 4b with the same responses and colors as Fig. 4a."""

from pathlib import Path
import shutil
import subprocess

from fig4_common import CACHE, use_private_base

plot_rastermap = use_private_base().plot_rastermap


OUT = Path(__file__).resolve().parent
GENERATED = CACHE / "figs/map_Beryl_cv_1_zsc_1_nclus_rm_100_sort_acs.svg"  # written by plot_rastermap into fig4/cache


def main() -> None:
    plot_rastermap(
        vers="concat",
        feat="concat_z",
        mapping="Beryl",
        sort_method="acs",
        bg=True,
        bg_bright=0.99,
        img_only=True,
        interp="antialiased",
        cv=True,
        zsc=True,
        vmax=2,
        bounds=False,
        rerun=False,
        clsfig=True,
    )
    if not GENERATED.is_file():
        raise FileNotFoundError(f"Expected output was not created: {GENERATED}")
    destination = OUT / "panel_b_anatomical_order_beryl_background.svg"
    shutil.copyfile(GENERATED, destination)
    subprocess.run([
        "inkscape", str(destination), "--export-type=pdf",
        f"--export-filename={destination.with_suffix('.pdf')}",
    ], check=True)
    subprocess.run([
        "inkscape", str(destination), "--export-type=png", "--export-dpi=360",
        f"--export-filename={destination.with_suffix('.png')}",
    ], check=True)
    print(f"Saved {destination.with_suffix('.*')} (SVG/PDF/PNG)")


if __name__ == "__main__":
    main()

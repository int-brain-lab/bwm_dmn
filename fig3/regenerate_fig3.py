#!/usr/bin/env python3
"""Recreate Figure 3 (Figure 4 before the 2026-10-06 reorder) from the version in the compressed-printer manuscript.

In that manuscript Figure 4 is the anatomy/function correspondence figure
(panels a--j). Its descriptive manuscript asset is
``anatomy_function_correspondence.pdf``. That asset is an Illustrator-assembled
PDF, so it is the highest-fidelity source for the complete multipanel layout.
This script makes a clean, deterministic copy and renders review formats without
rasterizing the PDF itself.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = (
    Path.home()
    / "dmn/manuscript/IBL_supersession_paper_resubmission"
    / "figures/anatomy_function_correspondence.pdf"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_pdf = args.output_dir / "fig3_regenerated.pdf"
    output_png_base = args.output_dir / "fig3_regenerated"
    output_svg = args.output_dir / "fig3_regenerated.svg"

    if not args.source.is_file():
        raise FileNotFoundError(args.source)

    # Preserve the original vector/raster mixture and Illustrator layout.
    shutil.copyfile(args.source, output_pdf)

    # Review copies; the PDF above remains the publication master.
    run([
        "pdftocairo", "-png", "-singlefile", "-r", "600",
        str(output_pdf), str(output_png_base),
    ])
    run(["pdftocairo", "-svg", str(output_pdf), str(output_svg)])

    if sha256(args.source) != sha256(output_pdf):
        raise RuntimeError("The regenerated PDF does not match its source asset")

    print(f"Source: {args.source}")
    print(f"Saved:  {output_pdf}")
    print(f"Saved:  {output_png_base.with_suffix('.png')}")
    print(f"Saved:  {output_svg}")
    print(f"SHA256: {sha256(output_pdf)}")


if __name__ == "__main__":
    main()

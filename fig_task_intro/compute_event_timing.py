#!/usr/bin/env python3
"""Average trial timing for Fig. 1b: stimulus onset -> movement onset -> feedback.

As dmn_bwm._build_event_stats (whose cache mean_event_diffs.npy no longer exists,
and which uses undefined names r and evs): for every session of the 515
insertions (pipeline/insertions.csv), the BWM trials with the BWM trial mask,
without the last 100 trials, mean delays stimOn -> firstMovement -> feedback;
then mean (and s.d.) over sessions. Needs ONE/Alyx access. Writes
cache/event_timing.npy.
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import dmn_bwm as d  # noqa: E402
from brainwidemap import load_trials_and_mask  # noqa: E402

EVENTS = ["stimOn_times", "firstMovement_times", "feedback_times"]
OUT = HERE / "cache" / "event_timing.npy"


def main():
    one = d.use_online_one()
    with open(HERE.parent / "pipeline" / "insertions.csv") as fh:
        eids = sorted({row["eid"] for row in csv.DictReader(fh)})
    diffs, failed = [], []
    for k, eid in enumerate(eids):
        try:
            trials, mask = load_trials_and_mask(one, eid)
            tr = trials[mask][:-100]
            diffs.append(np.nanmean(np.diff(tr[EVENTS].values, axis=1), axis=0))
        except Exception as e:  # noqa: BLE001
            failed.append((eid, repr(e)[:80]))
        if k % 50 == 0:
            print(f"{k}/{len(eids)}", flush=True)
    diffs = np.array(diffs)
    res = {"events": EVENTS, "diffs": diffs, "mean": np.nanmean(diffs, 0), "std": np.nanstd(diffs, 0),
           "n_sessions": len(diffs), "failed": failed}
    OUT.parent.mkdir(exist_ok=True)
    np.save(OUT, res, allow_pickle=True)
    print(f"{len(diffs)} sessions ({len(failed)} failed): stim->move {res['mean'][0]:.3f} s, "
          f"move->feedback {res['mean'][1]:.3f} s; saved {OUT}")


if __name__ == "__main__":
    main()

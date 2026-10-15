r"""Score a rubric session against the demonstrations, stage by stage.

Reads datasets\rollouts\<session>\trials.csv (written by rubric_session.ps1),
computes rubric_metrics.py columns for every trial, adds an aim reading from the
wrist camera, does the same for a sample of demonstrations, and writes one JSON
with per-trial rows, demo baselines and Clopper-Pearson intervals per stage.

Aim reading. aimed_at.py picks its own closure frame from the command trace and
looks at that exact frame, by which time a held cube can already fill or block the
view. This reads the frame LEAD frames BEFORE the first closure found by
rubric_metrics.closures, when the fingers are around the cube but not yet on it.
It is scored on demonstrations first, where the named cube is the answer, and the
hit rate is printed -- believe rollout readings only as far as that number allows.

    .\.venv-win\Scripts\python.exe tools\rollout\rubric_report.py s01
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "rollout"))
sys.path.insert(0, str(ROOT / "tools" / "dataset"))
import rubric_metrics as R  # noqa: E402
import box_cubes as B  # noqa: E402

GRIP_XY = (0.50, 0.72)   # where the fingers meet in the wrist view (aimed_at.py)
LEAD = 6                 # frames before the closure: 0.2 s
HOLD_THRESH = 1.5        # measured on so101_color6, see rubric_metrics.py
DEMO_SAMPLE = 60
EXCLUDE = {324, 354, 393, 402}


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    from math import comb

    def cdf(x, p):
        return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(x + 1))

    def solve(f, lo=0.0, hi=1.0):
        for _ in range(60):
            mid = (lo + hi) / 2
            if f(mid):
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(lambda p: 1 - cdf(k - 1, p) >= alpha / 2)
    upper = 1.0 if k == n else solve(lambda p: cdf(k, p) <= alpha / 2)
    return lower, upper


def aim(ds: Path, info: dict, ep_row, state, action):
    """(colour, distance in cube widths) of the cube nearest the fingers."""
    cl = R.closures(action[:, R.GRIP], state[:, R.GRIP])
    if not cl:
        return None, None, None
    f = max(cl[0][0] - LEAD, 0)
    B.DS = ds
    rgb = B.frame_at(info, ep_row, "observation.images.wrist", f / info["fps"])
    h, w = rgb.shape[:2]
    anchor = np.array([GRIP_XY[0] * w, GRIP_XY[1] * h])
    found = B.find(rgb, B.WRIST_AREA)
    if not found:
        return None, None, f
    scale = float(np.median([d["box"][2] for d in found]))
    best = min(found, key=lambda d: np.linalg.norm(np.asarray(d["centre"]) - anchor))
    dist = float(np.linalg.norm(np.asarray(best["centre"]) - anchor)) / scale
    return best["colour"], dist, f


def episodes(ds: Path):
    info = json.loads((ds / "meta" / "info.json").read_text(encoding="utf-8"))
    meta = pd.concat([pd.read_parquet(f) for f in sorted((ds / "meta" / "episodes").rglob("*.parquet"))],
                     ignore_index=True).set_index("episode_index")
    tasks = pd.read_parquet(ds / "meta" / "tasks.parquet")
    return info, meta, R.load(ds), {int(v): k for k, v in tasks["task_index"].items()}


def score(ds: Path, eps_wanted=None, named_override=None):
    info, meta, eps, _ = episodes(ds)
    rows = []
    for e, (st, ac) in sorted(eps.items()):
        if eps_wanted is not None and e not in eps_wanted:
            continue
        m = R.episode_metrics(st, ac, HOLD_THRESH)
        task = meta.loc[e]["tasks"][0]
        named = named_override or task.split()[2]
        colour, dist, frame = aim(ds, info, meta.loc[e], st, ac)
        m.update(episode=e, named=named, aimed=colour, aim_widths=dist, aim_frame=frame)
        rows.append(m)
    return rows


def main(argv) -> int:
    session = argv[0] if argv else "s01"
    sdir = ROOT / "datasets" / "rollouts" / session

    demo_ds = ROOT / "datasets" / "so101_color6"
    all_eps = sorted(set(R.load(demo_ds)) - EXCLUDE)
    pick = set(all_eps[:: max(1, len(all_eps) // DEMO_SAMPLE)][:DEMO_SAMPLE])
    demo = score(demo_ds, pick)
    readable = [r for r in demo if r["aimed"] and r["aim_widths"] <= 1.0]
    hits = sum(r["aimed"] == r["named"] for r in readable)
    print(f"demonstrations: {len(demo)} scored; aim readable (nearest cube within 1 width) "
          f"{len(readable)}/{len(demo)}, of those on the named cube {hits}/{len(readable)}")

    demo_metrics = pd.DataFrame([R.episode_metrics(st, ac, HOLD_THRESH)
                                 for e, (st, ac) in R.load(demo_ds).items() if e not in EXCLUDE])

    trials = []
    with open(sdir / "trials.csv", newline="") as fh:
        for t in csv.DictReader(fh):
            d = ROOT / t["dir"].replace("\\", "/")
            if not (d / "meta" / "info.json").exists():
                continue
            named = None if t["model"] == "colour" else "any"
            r = score(d, named_override=named)[0]
            r.update(n=int(t["n"]), model=t["model"], colour=t["colour"])
            trials.append(r)
    df = pd.DataFrame(trials).sort_values("n")

    def stages(g, model):
        n = len(g)
        near = g["aim_widths"].fillna(99) <= 1.0
        on_target = near & ((g["aimed"] == g["named"]) if model == "colour" else True)
        out = {
            "started": int((g["onset_s"] < 5).sum()),
            "closed": int((g["closes"] > 0).sum()),
            "aimed_at_a_cube": int(near.sum()),
            "aimed_at_target": int(on_target.sum()),
            "held_something": int((g["holds"] > 0).sum()),
        }
        return {k: {"k": v, "n": n, "ci": clopper_pearson(v, n)} for k, v in out.items()}

    summary = {}
    for model, g in df.groupby("model"):
        summary[model] = {
            "n": len(g),
            "stages": stages(g, model),
            "median": g[["onset_s", "reach_s", "closes", "holds", "sparc_reach", "reversals_hz",
                         "home_returns", "aim_widths"]].median().round(3).to_dict(),
        }
    summary["demo"] = {
        "n": len(demo_metrics),
        "median": demo_metrics[["onset_s", "reach_s", "closes", "holds", "sparc_reach",
                                "reversals_hz", "home_returns"]].median().round(3).to_dict(),
        "p10": demo_metrics[["sparc_reach", "reversals_hz"]].quantile(.1).round(3).to_dict(),
        "p90": demo_metrics[["sparc_reach", "reversals_hz"]].quantile(.9).round(3).to_dict(),
        "aim_validation": {"scored": len(demo), "readable": len(readable), "on_named": hits},
        "aim_widths_median": float(np.median([r["aim_widths"] for r in readable])),
    }

    out = ROOT / "logs" / "rubric" / f"{session}_report.json"
    out.write_text(json.dumps({"summary": summary,
                               "trials": df.replace({np.nan: None}).to_dict("records"),
                               "demo_sample": demo,
                               "demo_all": {"sparc_reach": demo_metrics["sparc_reach"].dropna().round(3).tolist(),
                                            "reversals_hz": demo_metrics["reversals_hz"].round(3).tolist()}},
                              default=float, indent=1), encoding="utf-8")
    cols = ["n", "model", "colour", "onset_s", "closes", "holds", "home_returns",
            "aimed", "aim_widths", "sparc_reach", "reversals_hz"]
    print(df[cols].round(2).to_string(index=False))
    for model in ("colour", "pooled"):
        if model in summary:
            s = summary[model]["stages"]
            print(f"\n{model} (n={summary[model]['n']}): " +
                  "  ".join(f"{k} {v['k']}/{v['n']}" for k, v in s.items()))
    print(f"\nwritten {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

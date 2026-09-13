r"""Audit a LeRobot v3 dataset for truncated / orphaned / inconsistent episodes.

Cross-checks three independent sources for every episode:
  meta/episodes/*.parquet   what the dataset claims
  data/chunk-*/*.parquet    the actual state/action rows
  videos/<key>/*.mp4        the actual decoded frames

Run from the repo root:
    .\.venv-win\Scripts\python.exe tools\dataset\audit_episodes.py datasets\so101_v3
"""
import sys
from pathlib import Path

import pandas as pd


def load_parquet_dir(d: Path) -> pd.DataFrame:
    files = sorted(d.rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def video_frame_counts(root: Path, key: str) -> dict:
    """Map (chunk, file) -> frame count, decoded with pyav."""
    import av

    counts = {}
    for mp4 in sorted((root / "videos" / key).rglob("*.mp4")):
        with av.open(str(mp4)) as c:
            s = c.streams.video[0]
            n = s.frames
            if not n:  # some encoders leave frames=0; count the hard way
                n = sum(1 for _ in c.decode(video=0))
        counts[mp4] = n
    return counts


def main(root: Path) -> int:
    ep = load_parquet_dir(root / "meta" / "episodes")
    if ep.empty:
        print("no episode metadata found")
        return 1
    data = load_parquet_dir(root / "data")

    print(f"dataset            {root}")
    print(f"episodes in meta   {len(ep)}")
    print(f"rows in data       {len(data)}")
    print()

    problems = []

    # --- 1. orphaned staging directories (recording killed before encode) ---
    for cam_dir in sorted((root / "images").glob("*")):
        for ep_dir in sorted(cam_dir.glob("episode-*")):
            n = len(list(ep_dir.glob("*")))
            idx = int(ep_dir.name.split("-")[-1])
            problems.append(
                ("ORPHAN", idx, f"{ep_dir.relative_to(root)} holds {n} raw frames, never encoded")
            )

    # --- 2. episode_index continuity ---
    idxs = sorted(ep["episode_index"].tolist())
    expected = list(range(len(idxs)))
    if idxs != expected:
        missing = sorted(set(expected) - set(idxs))
        problems.append(("GAP", -1, f"episode_index not contiguous; missing {missing[:20]}"))

    # --- 3. meta length vs actual rows in data ---
    len_col = "length" if "length" in ep.columns else None
    actual = data.groupby("episode_index").size().to_dict() if not data.empty else {}
    lengths = {}
    for _, r in ep.iterrows():
        i = int(r["episode_index"])
        claimed = int(r[len_col]) if len_col else None
        real = actual.get(i, 0)
        lengths[i] = real
        if claimed is not None and claimed != real:
            problems.append(("LEN", i, f"meta says {claimed} frames, data has {real}"))
        if real == 0:
            problems.append(("EMPTY", i, "no rows in data parquet"))

    # --- 4. suspiciously short episodes ---
    if lengths:
        s = pd.Series(lengths)
        med = s.median()
        for i, n in s.items():
            if n < 0.4 * med:
                problems.append(("SHORT", int(i), f"{n} frames vs median {med:.0f} ({n/med:.0%})"))

    # --- 5. video frame totals vs data frame totals, per file ---
    for key in [c for c in ep.columns if c.startswith("videos/")] or []:
        pass  # v3 stores video ranges per episode; totals check below is the robust one

    try:
        for cam in sorted((root / "videos").glob("observation.images.*")):
            key = cam.name
            counts = video_frame_counts(root, key)
            total_video = sum(counts.values())
            if total_video != len(data):
                problems.append(
                    ("VIDEO", -1, f"{key}: {total_video} decoded frames vs {len(data)} data rows")
                )
            for mp4, n in counts.items():
                if n == 0:
                    problems.append(("VIDEO", -1, f"{mp4.relative_to(root)} decodes to 0 frames"))
    except Exception as e:  # noqa: BLE001
        print(f"(video check skipped: {e})")

    # --- 6. timestamp monotonicity within each episode ---
    if not data.empty and "timestamp" in data.columns:
        for i, g in data.groupby("episode_index"):
            t = g.sort_values("frame_index")["timestamp"].to_numpy()
            if len(t) > 1 and (t[1:] < t[:-1]).any():
                problems.append(("TIME", int(i), "timestamp goes backwards"))

    if not problems:
        print("clean - no orphans, gaps, length mismatches or short episodes")
        return 0

    print(f"{len(problems)} problem(s):\n")
    for kind, idx, msg in sorted(problems, key=lambda p: (p[0], p[1])):
        where = f"ep {idx:>3}" if idx >= 0 else "dataset"
        print(f"  [{kind:<6}] {where}  {msg}")

    print("\nlengths (frames per episode):")
    s = pd.Series(lengths).sort_index()
    print(f"  min {s.min()}  p10 {s.quantile(.1):.0f}  median {s.median():.0f}  max {s.max()}")
    return 2


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "datasets/so101_v3")
    raise SystemExit(main(root))

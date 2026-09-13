r"""Close gaps in a LeRobot v3 dataset's episode_index and resync meta/info.json.

A crash or a re-record during `lerobot-record` can leave the episode counter ahead
of what was actually written, producing a non-contiguous episode_index. LeRobot's
reader then computes requested = set(range(total_episodes)), finds it is not a
subset of what is on disk, declares the local copy incomplete and tries to fetch
the dataset from the Hub.

Everything else in the v3 metadata is order-based (dataset_from_index,
dataset_to_index, video from_timestamp/to_timestamp, file_index), so the repair is
a pure relabel of episode_index plus a corrected info.json. No data is moved and no
video is re-encoded.

    .\.venv-win\Scripts\python.exe tools\dataset\renumber_episodes.py datasets\so101_v3
    .\.venv-win\Scripts\python.exe tools\dataset\renumber_episodes.py datasets\so101_v3 --apply
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def episode_parquets(root: Path) -> list[Path]:
    return sorted((root / "meta" / "episodes").rglob("*.parquet"))


def data_parquets(root: Path) -> list[Path]:
    return sorted((root / "data").rglob("*.parquet"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    ap.add_argument("--drop-staging", action="store_true", help="also delete orphaned images/ dirs")
    args = ap.parse_args()
    root = args.root

    meta_files = episode_parquets(root)
    ep = pd.concat([pd.read_parquet(f) for f in meta_files], ignore_index=True)
    old = sorted(int(i) for i in ep["episode_index"])
    remap = {o: n for n, o in enumerate(old)}
    moved = {o: n for o, n in remap.items() if o != n}

    total_frames = int(ep["length"].sum())
    print(f"episodes on disk   {len(old)}")
    print(f"total frames       {total_frames}")
    if not moved:
        print("episode_index is already contiguous")
    else:
        lo = min(moved)
        print(f"renumbering        {len(moved)} episodes, first change {lo} -> {moved[lo]}")
        print(f"                   {sorted(moved.items())[:10]}{' ...' if len(moved) > 10 else ''}")

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    print(f"info.json says     {info['total_episodes']} episodes / {info['total_frames']} frames")
    print(f"correcting to      {len(old)} episodes / {total_frames} frames")

    staging = [d for c in (root / "images").glob("*") for d in c.glob("episode-*")] if (root / "images").exists() else []
    for d in staging:
        print(f"orphan staging     {d.relative_to(root)} ({len(list(d.glob('*')))} raw frames)")

    if not args.apply:
        print("\ndry run - nothing written. re-run with --apply")
        return 0

    # --- meta/episodes: episode_index and the per-episode stats of that column ---
    stat_cols = [c for c in ep.columns if c.startswith("stats/episode_index/")]
    for f in meta_files:
        df = pd.read_parquet(f)
        df["episode_index"] = df["episode_index"].map(remap).astype(df["episode_index"].dtype)
        for c in stat_cols:
            suffix = c.rsplit("/", 1)[1]
            if suffix in ("min", "max", "q01", "q10", "q50", "q90", "q99", "mean"):
                df[c] = df["episode_index"].astype(float).map(lambda v, _c=c: [v])
            elif suffix == "std":
                df[c] = df[c].map(lambda _v: [0.0])
        df.to_parquet(f, index=False)
        print(f"wrote {f.relative_to(root)}")

    # --- data: episode_index column ---
    for f in data_parquets(root):
        df = pd.read_parquet(f)
        if not df["episode_index"].isin(remap).all():
            raise SystemExit(f"{f} holds episode indices absent from meta - aborting")
        df["episode_index"] = df["episode_index"].map(remap).astype(df["episode_index"].dtype)
        df.to_parquet(f, index=False)
        print(f"wrote {f.relative_to(root)}")

    # --- info.json ---
    info["total_episodes"] = len(old)
    info["total_frames"] = total_frames
    info["splits"] = {"train": f"0:{len(old)}"}
    info_path.write_text(json.dumps(info, indent=4))
    print(f"wrote {info_path.relative_to(root)}")

    # --- stats.json: global episode_index range ---
    sp = root / "meta" / "stats.json"
    if sp.exists():
        st = json.loads(sp.read_text())
        if "episode_index" in st:
            st["episode_index"]["min"] = [0.0]
            st["episode_index"]["max"] = [float(len(old) - 1)]
            sp.write_text(json.dumps(st, indent=4))
            print(f"wrote {sp.relative_to(root)}")

    if args.drop_staging:
        for d in staging:
            shutil.rmtree(d)
            print(f"deleted {d.relative_to(root)}")

    print("\ndone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

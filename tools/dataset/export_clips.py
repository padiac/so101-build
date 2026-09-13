r"""Cut a LeRobot v3 dataset into one reviewable mp4 per episode.

Episodes are packed many-to-a-file in v3, so there is nothing to double-click.
This slices each episode out using the from/to timestamps in meta/episodes and
writes a single h264 clip with the cameras side by side, burned-in episode index,
frame number and gripper value, so a bad take can be spotted and its number read
straight off the picture.

    .\.venv-win\Scripts\python.exe tools\dataset\export_clips.py datasets\so101_v3 --episodes 75-81
    .\.venv-win\Scripts\python.exe tools\dataset\export_clips.py datasets\so101_v3 --episodes all --scale 0.4
"""
import argparse
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def parse_episodes(spec: str, available: list[int]) -> list[int]:
    if spec == "all":
        return available
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    missing = [e for e in out if e not in available]
    if missing:
        raise SystemExit(f"episodes not in dataset: {missing}")
    return out


def decode_segment(path: Path, t0: float, t1: float) -> list[np.ndarray]:
    """Decode frames with t0 <= pts < t1, seeking to just before t0."""
    frames = []
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        seek_t = max(t0 - 1.0, 0.0)
        c.seek(int(seek_t / s.time_base), stream=s)
        for f in c.decode(video=0):
            t = float(f.pts * s.time_base)
            if t < t0 - 1e-6:
                continue
            if t >= t1 - 1e-6:
                break
            frames.append(f.to_ndarray(format="rgb24"))
    return frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--episodes", default="all", help='e.g. "75-81", "3,7,12", "all"')
    ap.add_argument("--out", type=Path, default=None, help="default <root>/clips")
    ap.add_argument("--scale", type=float, default=0.5, help="resize factor per camera")
    ap.add_argument("--fps", type=int, default=None, help="playback fps (default: dataset fps)")
    args = ap.parse_args()

    root = args.root
    out = args.out or (root / "clips")
    out.mkdir(parents=True, exist_ok=True)

    ep = pd.concat(
        [pd.read_parquet(f) for f in sorted((root / "meta" / "episodes").rglob("*.parquet"))],
        ignore_index=True,
    ).set_index("episode_index")
    data = pd.concat(
        [pd.read_parquet(f) for f in sorted((root / "data").rglob("*.parquet"))], ignore_index=True
    )

    import json

    info = json.loads((root / "meta" / "info.json").read_text())
    fps = args.fps or int(info["fps"])
    vid_keys = [k for k in info["features"] if info["features"][k]["dtype"] == "video"]

    todo = parse_episodes(args.episodes, sorted(int(i) for i in ep.index))
    print(f"{len(todo)} episode(s) -> {out}\n")

    for e in todo:
        row = ep.loc[e]
        streams = []
        for k in vid_keys:
            vpath = root / info["video_path"].format(
                video_key=k,
                chunk_index=int(row[f"videos/{k}/chunk_index"]),
                file_index=int(row[f"videos/{k}/file_index"]),
            )
            t0 = float(row[f"videos/{k}/from_timestamp"])
            t1 = float(row[f"videos/{k}/to_timestamp"])
            streams.append((k, decode_segment(vpath, t0, t1)))

        n = min(len(f) for _, f in streams)
        if n == 0:
            print(f"  ep {e:>3}  no frames decoded - skipped")
            continue

        g = data[data.episode_index == e].sort_values("frame_index")
        grip = np.stack(g["observation.state"].to_numpy())[:, 5]

        h0, w0 = streams[0][1][0].shape[:2]
        w, h = int(w0 * args.scale), int(h0 * args.scale)
        W, H = w * len(streams), h + 22

        dst = out / f"episode-{e:03d}.mp4"
        with av.open(str(dst), "w") as oc:
            st = oc.add_stream("libx264", rate=fps)
            st.width, st.height, st.pix_fmt = W, H, "yuv420p"
            st.options = {"crf": "23", "preset": "veryfast"}
            st.time_base = Fraction(1, fps)
            for i in range(n):
                canvas = Image.new("RGB", (W, H), (16, 16, 16))
                for j, (_k, frames) in enumerate(streams):
                    canvas.paste(Image.fromarray(frames[i]).resize((w, h)), (j * w, 22))
                d = ImageDraw.Draw(canvas)
                gv = grip[i] if i < len(grip) else float("nan")
                d.text((6, 5), f"ep {e}   frame {i}/{n}   {i / fps:5.2f}s   gripper {gv:5.2f}", fill=(255, 230, 90))
                for j, (k, _f) in enumerate(streams):
                    d.text((j * w + 6, 26), k.rsplit(".", 1)[-1], fill=(120, 220, 255))
                for p in st.encode(av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")):
                    oc.mux(p)
            for p in st.encode():
                oc.mux(p)

        print(f"  ep {e:>3}  {n:>4} frames  {n / fps:5.1f}s  ->  {dst.relative_to(root.parent)}")

    print(f"\nopen the folder:  explorer {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

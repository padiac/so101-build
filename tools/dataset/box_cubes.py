r"""Draw a box round every cube in a frame, so what the model is given can be seen.

The policy has no detection head and never outputs a box -- nothing here reads
the model's mind. This answers a narrower question: in the picture the model
actually receives, are the six cubes there and are the colours separable at all?

The hue bands are not guessed. They come from the histogram of strongly
saturated pixels across real frames, which falls into six clean clusters. Pink
sits below the saturation cut because the cube is pale, and it is the one colour
no instruction ever names, so it gets its own looser band and is drawn as a
distractor.

Run it and look at the pictures. A detector trusted without looking is how the
grasp landmark went wrong twice.

    .\.venv-win\Scripts\python.exe tools\dataset\box_cubes.py OUTDIR [--wrist]
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "datasets" / "so101_color6"

# (name, hue spans, min saturation, max saturation, min value, draw colour BGR).
# Red wraps past 180 and is given as two bands.
#
# Saturation carries as much of the separation as hue does.
#
# The floors differ because the cubes are not equally vivid: on one frame holding
# all six, blue sits at S 177 but purple at 68 and pink at 26. A single floor at
# 90 found four of them and looked fine until the pictures were opened.
#
# The ceilings exist because blue and purple are not separable by hue at all. Under
# the top camera purple reads H 130, but a hand's width away under the wrist camera
# the same cube reads H 119 -- inside blue's band. Splitting them on hue dropped
# purple from the very frame where the gripper was closing on it. Vividness is the
# difference that actually holds: blue is saturated, purple is pale.
BANDS = [
    ("red",    [(0, 10), (160, 180)],  90, 255,  60, (60, 60, 220)),
    ("yellow", [(15, 32)],             90, 255,  60, (60, 210, 230)),
    ("green",  [(68, 95)],             90, 255,  60, (90, 200, 90)),
    ("blue",   [(100, 125)],          110, 255,  60, (220, 140, 60)),
    ("purple", [(112, 152)],           35, 110,  90, (200, 90, 170)),
    ("pink",   [(0, 12), (152, 180)],  18, 110, 150, (190, 170, 235)),
]
MIN_AREA = 250          # a cube face is about 44x44 px in the top view
# The desk holds a pink keyboard, which pink's low floor would otherwise claim as
# one enormous cube. Nothing on the bowl is more than a few cube faces across.
MAX_AREA = 6000
# The wrist camera is a hand's width from the cubes at the grasp, where one face
# fills a fifth of the frame. The top view's ceiling silently rejected every cube
# in those frames and left only a sliver of a distant one, which read as the arm
# aiming somewhere it was not.
WRIST_AREA = (800, 150000)


def find(rgb: np.ndarray, area: tuple[int, int] | None = None) -> list[dict]:
    """Every cube-sized blob of each colour, largest first.

    `area` is the (min, max) pixel area a blob may have; it depends on how
    far the camera is from the cubes, so the caller says which view this is.
    """
    amin, amax = area or (MIN_AREA, MAX_AREA)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    claimed = np.zeros(hsv.shape[:2], bool)
    out = []
    for name, spans, smin, smax, vmin, bgr in BANDS:
        m = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in spans:
            m |= cv2.inRange(hsv, (lo, smin, vmin), (hi, smax, 255))
        # Pink overlaps red's hue and is separated only by saturation, so it must
        # not re-claim pixels a stronger colour already took.
        m[claimed] = 0
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if not amin <= area <= amax:
                continue
            # A cube face is roughly square from above however it is tumbled.
            # This drops the slivers the desk furniture leaves at the edges.
            if not 0.5 <= w / max(h, 1) <= 2.0:
                continue
            claimed |= lab == i
            out.append({"colour": name, "box": (int(x), int(y), int(w), int(h)),
                        "centre": (float(cent[i][0]), float(cent[i][1])),
                        "area": int(area), "bgr": bgr})
    return out


def draw(rgb: np.ndarray, found: list[dict], named: str | None = None) -> np.ndarray:
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for f in found:
        x, y, w, h = f["box"]
        hit = f["colour"] == named
        cv2.rectangle(img, (x, y), (x + w, y + h), f["bgr"], 3 if hit else 1)
        tag = f["colour"].upper() if hit else f["colour"]
        cv2.putText(img, tag, (x, max(y - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, f["bgr"], 2 if hit else 1, cv2.LINE_AA)
    if named:
        cv2.putText(img, f"asked for: {named}", (8, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def frame_at(info, ep_row, key, t):
    import av

    v = DS / info["video_path"].format(
        video_key=key,
        chunk_index=int(ep_row[f"videos/{key}/chunk_index"]),
        file_index=int(ep_row[f"videos/{key}/file_index"]))
    t0 = float(ep_row[f"videos/{key}/from_timestamp"]) + t
    with av.open(str(v)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        c.seek(int(max(t0 - 0.5, 0) / s.time_base), stream=s)
        for f in c.decode(video=0):
            if float(f.pts * s.time_base) >= t0 - 1e-6:
                return f.to_ndarray(format="rgb24")
    raise RuntimeError("no frame")


def main(argv):
    if not argv:
        print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
        return 2
    out = Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    key = ("observation.images.wrist" if "--wrist" in argv
           else "observation.images.top")

    info = json.loads((DS / "meta" / "info.json").read_text(encoding="utf-8"))
    ep = pd.concat([pd.read_parquet(f) for f in
                    sorted((DS / "meta" / "episodes").rglob("*.parquet"))],
                   ignore_index=True).set_index("episode_index")
    data = pd.concat([pd.read_parquet(f) for f in
                      sorted((DS / "data").rglob("*.parquet"))], ignore_index=True)
    tasks = pd.read_parquet(DS / "meta" / "tasks.parquet")
    task_of = {int(v): k for k, v in tasks["task_index"].items()}

    want, counts = {}, []
    for e, r in data.groupby("episode_index"):
        want.setdefault(int(r["task_index"].iloc[0]), int(e))
    for ti, e in sorted(want.items()):
        named = task_of[ti].split()[2]
        rgb = frame_at(info, ep.loc[e], key, 0.0)
        found = find(rgb)
        p = out / f"ep{e:03d}_{named}.png"
        cv2.imwrite(str(p), draw(rgb, found, named))
        got = [f["colour"] for f in found]
        counts.append((e, named, got))
        print(f"ep {e:3d}  asked for {named:<7} found {len(found)}: "
              f"{', '.join(got)}   -> {p.name}")

    hit = sum(1 for _, n, g in counts if n in g)
    print(f"\nthe named colour was found in {hit}/{len(counts)} frames")
    print("Open the pictures and check the boxes before trusting any of this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""Full camera matrix: which indices stream, alone and in combination.

Context (measured 2026-08-29): all three cameras sit on the same Intel xHCI
controller and the same USB root hub. They are USB 2.0 devices, so they share
one 480 Mbps bus -- about 35-40 MB/s in practice. Uncompressed YUY2 at
640x480x30 costs 18.4 MB/s per camera, so two just fit and three cannot.

MJPG would cut that by roughly 10x, but lerobot logs
    failed to set fourcc=MJPG (actual=, success=False)
under the default backend, because MSMF ignores fourcc requests. DSHOW honours
them. That is why the backend column matters here.

Run this whenever the USB layout changes -- indices are reassigned on every
plug or unplug, and a camera reading black is far more often a renumbering than
a fault.

Usage:
    python cam_matrix.py
    python cam_matrix.py --indices 0 1 2
"""

import argparse
import itertools

import cv2
import numpy as np

BACKENDS = [("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW)]
DEAD = 5.0


def grab(indices, backend, fourcc=None, n=20):
    """Open the given indices together and report each one's mean pixel value."""
    caps = []
    try:
        for i in indices:
            c = cv2.VideoCapture(i, backend)
            if not c.isOpened():
                caps.append((i, None))
                continue
            if fourcc:
                c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            c.set(cv2.CAP_PROP_FPS, 30)
            caps.append((i, c))

        vals = {i: [] for i, _ in caps}
        for _ in range(n):
            for i, c in caps:
                if c is None:
                    continue
                ok, f = c.read()
                if ok:
                    vals[i].append(float(f.mean()))
        out = {}
        for i, c in caps:
            if c is None:
                out[i] = None
            else:
                v = vals[i]
                out[i] = float(np.mean(v)) if v else 0.0
        return out
    finally:
        for _, c in caps:
            if c is not None:
                c.release()


def fmt(res):
    parts = []
    for i in sorted(res):
        v = res[i]
        if v is None:
            parts.append("%d:open-fail" % i)
        else:
            parts.append("%d:%5.1f%s" % (i, v, "" if v > DEAD else " DEAD"))
    return "  ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--frames", type=int, default=20)
    a = ap.parse_args()
    idx = a.indices

    combos = []
    for r in range(1, len(idx) + 1):
        combos.extend(itertools.combinations(idx, r))

    for bname, backend in BACKENDS:
        for fourcc in (None, "MJPG"):
            tag = bname + ("+MJPG" if fourcc else "")
            print("=== %s ===" % tag)
            for combo in combos:
                res = grab(list(combo), backend, fourcc, a.frames)
                print("  %-12s %s" % ("+".join(map(str, combo)), fmt(res)))
            print()

    print("DEAD = mean pixel value below %.0f, i.e. a black feed." % DEAD)
    print("A row where each camera is fine alone but one dies in combination is")
    print("bandwidth, not a broken camera.")


if __name__ == "__main__":
    main()

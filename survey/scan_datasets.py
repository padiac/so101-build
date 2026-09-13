"""Scan community SO-101 datasets for co-training compatibility.

What actually has to match for a dataset to be mixable with ours:
  codebase v3.0, fps 30, state/action shape [6], identical joint names.
Cameras do NOT have to match -- SmolVLA pads missing views and masks them out.
"""
import io
import json
import sys
import time
import urllib.request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

OURS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
        "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]


def get(url, as_json=True):
    r = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    d = urllib.request.urlopen(r, timeout=45).read()
    return json.loads(d) if as_json else d


def api(p):
    return get("https://huggingface.co/api/" + p)


cands = {}
for q in ["so101", "so100", "so-101", "so101 pick", "so101 cube", "so101 color", "so101 sort"]:
    try:
        for d in api(f"datasets?search={urllib.parse.quote(q) if False else q}&limit=100&sort=downloads&direction=-1"):
            cands[d["id"]] = d.get("downloads", 0)
    except Exception:
        pass
    time.sleep(0.3)
print(f"候选 {len(cands)} 个，按下载量检查前 90 个\n", flush=True)

rows, checked = [], 0
for rid, dl in sorted(cands.items(), key=lambda x: -x[1]):
    if checked >= 90:
        break
    checked += 1
    try:
        info = get(f"https://huggingface.co/datasets/{rid}/resolve/main/meta/info.json")
    except Exception:
        continue
    try:
        f = info["features"]
        names = (f.get("action") or {}).get("names") or []
        if isinstance(names, dict):
            names = names.get("motors") or []
        ok = (info.get("codebase_version") == "v3.0"
              and info.get("fps") == 30
              and (f.get("action") or {}).get("shape") == [6]
              and (f.get("observation.state") or {}).get("shape") == [6]
              and list(names) == OURS)
        cams = [k for k, v in f.items() if v.get("dtype") == "video"]
        task = ""
        try:
            import pandas as pd
            tp = get(f"https://huggingface.co/datasets/{rid}/resolve/main/meta/tasks.parquet", as_json=False)
            task = "; ".join(pd.read_parquet(io.BytesIO(tp)).index.astype(str)[:3])[:70]
        except Exception:
            pass
        rows.append(dict(id=rid, dl=dl, ok=ok, eps=info.get("total_episodes", 0),
                         frames=info.get("total_frames", 0), ncam=len(cams),
                         cams=",".join(c.split(".")[-1] for c in cams)[:34],
                         ver=info.get("codebase_version"), fps=info.get("fps"), task=task))
    except Exception:
        continue
    time.sleep(0.15)

good = [r for r in rows if r["ok"]]
print(f"能读到 meta 的 {len(rows)} 个，其中 {len(good)} 个格式完全兼容\n")
print(f"{'集':>4} {'帧':>7} {'相机':>4}  {'ID':<52} 任务")
tot_e = tot_f = 0
for r in sorted(good, key=lambda x: -x["frames"])[:35]:
    tot_e += r["eps"]; tot_f += r["frames"]
    print(f"{r['eps']:>4} {r['frames']:>7} {r['ncam']:>4}  {r['id']:<52} {r['task']}")
print(f"\n前 35 个合计 {tot_e} 集 / {tot_f} 帧")

bad = [r for r in rows if not r["ok"]]
print(f"\n不兼容的 {len(bad)} 个，原因分布：")
import collections
c = collections.Counter()
for r in bad:
    if r["ver"] != "v3.0": c[f"格式 {r['ver']}"] += 1
    elif r["fps"] != 30: c[f"fps {r['fps']}"] += 1
    else: c["关节名或维度不符"] += 1
for k, v in c.most_common(): print(f"   {v:>3}  {k}")

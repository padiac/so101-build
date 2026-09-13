r"""What each checkpoint in policies/ actually is, read from the checkpoint.

A hand-written list of models goes stale the first time someone trains one and
forgets to edit it, and a stale list is worse than none: it is confidently wrong
about which weights are which. Everything here comes out of the checkpoint's own
config.json and train_config.json, so it cannot disagree with what is on disk.

MODELS.md holds the part that is NOT in the checkpoint -- what each dataset was
recorded doing, and which models are worth running. This prints the rest.

    .\.venv-win\Scripts\python.exe tools\policy\list_models.py
    .\.venv-win\Scripts\python.exe tools\policy\list_models.py --md
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# What each dataset was recorded doing. The checkpoint records which dataset it
# used but not what the person was doing at the time, and that is the thing you
# actually want to know a year later.
DATASETS = {
    "so101_pickplace": "yellow block into the bowl, the first 30 takes",
    "so101_v2":        "yellow block into the bowl, 75 takes",
    "so101_v3":        "yellow block into the bowl, 130 takes, varied positions",
    "so101_color6":    "named cube OUT of the bowl, 5 colours x 30, pink held out",
    "so101_mix1":      "so101_v3 + so101_color6, both directions, 280 takes",
    "so101_c6pool":    "so101_color6 with the 5 instructions pooled into one, "
                       "150 takes -- any cube counts",
    "so101_color6_304": "named cube OUT of the bowl, 304 takes, 5 colours x ~60, "
                        "pink held out",
    "so101_c6pool_304": "so101_color6_304 with the 5 instructions pooled into "
                        "one, 304 takes -- any cube counts",
}


def facts(d: Path) -> dict | None:
    cf, tc = d / "config.json", d / "train_config.json"
    if not cf.exists():
        return None
    c = json.loads(cf.read_text(encoding="utf-8"))
    r = {
        "name": d.name,
        "type": c.get("type", "?"),
        "chunk": c.get("chunk_size"),
        "cams": ",".join(sorted(k.split(".")[-1] for k in c.get("input_features", {})
                                if ".images." in k)),
        "dataset": "?", "steps": "?", "batch": "?", "base": "",
    }
    if tc.exists():
        t = json.loads(tc.read_text(encoding="utf-8"))
        ds = (t.get("dataset") or {}).get("repo_id", "?")
        # local/so101_mix1_trim -> so101_mix1. The trim suffix is a processing
        # step, not a different recording.
        r["dataset"] = ds.split("/")[-1].removesuffix("_trim").removesuffix("_nostate")
        r["steps"] = t.get("steps", "?")
        r["batch"] = t.get("batch_size", "?")
        base = (t.get("policy") or {}).get("pretrained_path") or ""
        r["base"] = "smolvla_base" if "smolvla_base" in str(base) else (
            "resumed" if base else "from scratch")
    w = d / "model.safetensors"
    r["mb"] = round(w.stat().st_size / 1e6) if w.exists() else 0
    r["when"] = (datetime.fromtimestamp(w.stat().st_mtime).strftime("%Y-%m-%d")
                 if w.exists() else "?")
    r["recorded"] = DATASETS.get(r["dataset"], "unknown dataset")
    # ACT has no language input at all: task, language and text appear once in
    # its model file, inside the licence. Whatever you type, it does the one
    # thing its dataset contains. This is the single most useful thing to know
    # before spending an evening testing instructions against one.
    r["language"] = "no" if r["type"] == "act" else "yes"
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true", help="markdown table for MODELS.md")
    args = ap.parse_args()

    rows = [f for d in sorted((ROOT / "policies").iterdir()) if d.is_dir()
            for f in [facts(d)] if f]
    if not rows:
        print("no checkpoints in policies/")
        return 1

    if args.md:
        print("| checkpoint | type | follows instructions | trained on | what that is |"
              " steps | cameras | date |")
        print("|---|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| `{r['name']}` | {r['type']} | {r['language']} | {r['dataset']} |"
                  f" {r['recorded']} | {r['steps']} | {r['cams']} | {r['when']} |")
        return 0

    w = max(len(r["name"]) for r in rows)
    print(f"{'checkpoint':<{w}}  {'type':<8}{'lang':<6}{'steps':>7}  "
          f"{'date':<11}{'cameras':<24}trained on")
    print("-" * (w + 76))
    for r in rows:
        print(f"{r['name']:<{w}}  {r['type']:<8}{r['language']:<6}{str(r['steps']):>7}  "
              f"{r['when']:<11}{r['cams']:<24}{r['dataset']}")
    print()
    for r in rows:
        print(f"{r['name']}\n    {r['recorded']}")
        if r["language"] == "no":
            print("    ignores the instruction entirely -- one dataset, one behaviour")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

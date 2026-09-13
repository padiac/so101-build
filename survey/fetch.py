"""Download the models and datasets worth having locally, from the 2026-09-03 survey."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from huggingface_hub import snapshot_download

MODELS = [
    ("lerobot/smolvla_base",        "计划的主力：预训练在 SO-100/101 社区数据上"),
    ("MINT-SJTU/Evo-1",             "轻量 VLA 3.1GB，InternVL3-1B 骨干"),
    ("lerobot/VLA-JEPA-Pretrain",   "JEPA 路线，和世界模型直接相关"),
    ("lerobot/xvla-base",           "3.5GB，另一个能装下的候选"),
]
DATASETS = [
    ("TommyZihao/lerobot_zihao_dataset_shake_hands", "真机 SO-101，LingBot 后训练用的那份"),
    ("TommyZihao/lerobot_zihao_dataset_a",           "真机 SO-101，夹放砂糖橘"),
    ("lerobot/svla_so101_pickplace",                 "官方 SO-101 pickplace 参考数据"),
]

for repo, why in MODELS:
    try:
        p = snapshot_download(repo, local_dir=f"survey/models/{repo.split('/')[-1]}")
        print(f"OK   模型 {repo}  ({why})")
    except Exception as e:
        print(f"FAIL 模型 {repo}: {type(e).__name__} {str(e)[:120]}")

for repo, why in DATASETS:
    try:
        p = snapshot_download(repo, repo_type="dataset", local_dir=f"survey/datasets/{repo.split('/')[-1]}")
        print(f"OK   数据 {repo}  ({why})")
    except Exception as e:
        print(f"FAIL 数据 {repo}: {type(e).__name__} {str(e)[:120]}")
print("ALL DONE")

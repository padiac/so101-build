"""离线验证策略：喂训练数据里的真实观测，比对预测动作和录制动作。

为什么要这一步：
    上机表现不好时，有两类完全不同的原因：
      A. 模型没学会 —— 预测本身就错
      B. 模型学会了，但部署这一环喂错了东西（相机、初始姿态、限幅…）
    这两类的修法毫无交集，必须先分开。

    喂数据集里的观测是**分布内**的输入，等于把部署环节整个摘掉。
    这里如果预测得准 -> 模型没问题，去查部署。
    这里如果预测得也烂 -> 别折腾硬件了，问题在训练/数据。

用法：
    python eval_offline.py [n_samples]
"""

import sys

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60

ds = LeRobotDataset("local/so101_pickplace", root="datasets/so101_v3")
CKPT = "policies/act_v3_100k"
policy = ACTPolicy.from_pretrained(CKPT)
policy.to("cpu")
policy.eval()

# lerobot 0.6 把归一化拆成了独立的 pre/post processor（checkpoint 里的
# policy_preprocessor_* / policy_postprocessor_* 文件）。lerobot-rollout 走的是
#     preprocessor -> policy -> postprocessor
# 少了这两步，喂进去的是未归一化的原始值，输出也没反归一化，
# 结果会烂得像模型没训练 —— 这是测法错，不是模型错。
cfg = PreTrainedConfig.from_pretrained(CKPT)
# 预处理管线里烘死了训练时的 device=cuda，本机推理在 CPU 上，必须覆盖。
# lerobot-rollout 也是这么做的（rollout/context.py:471）。
pre, post = make_pre_post_processors(
    cfg,
    pretrained_path=CKPT,
    preprocessor_overrides={"device_processor": {"device": "cpu"}},
)

rng = np.random.default_rng(0)
idxs = rng.choice(len(ds), size=N, replace=False)

errs, preds, gts = [], [], []
with torch.no_grad():
    for i in idxs:
        s = ds[int(i)]
        batch = {
            "observation.state": s["observation.state"][None],
            "observation.images.top": s["observation.images.top"][None],
            "observation.images.wrist": s["observation.images.wrist"][None],
        }
        policy.reset()                      # 强制真前向，不走动作队列
        a = post(policy.select_action(pre(batch)))[0]
        gt = s["action"]
        errs.append((a - gt).abs().numpy())
        preds.append(a.numpy())
        gts.append(gt.numpy())

errs = np.array(errs)
preds = np.array(preds)
gts = np.array(gts)

names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
print("样本 %d 个，随机抽自 %d 帧\n" % (N, len(ds)))
print("%-15s %9s %9s %9s %9s" % ("joint", "MAE", "动作范围", "MAE/范围", "评价"))
for j, nm in enumerate(names):
    mae = errs[:, j].mean()
    rng_j = gts[:, j].max() - gts[:, j].min()
    frac = mae / rng_j if rng_j > 1e-9 else float("nan")
    verdict = "好" if frac < 0.05 else ("一般" if frac < 0.15 else "差")
    print("%-15s %9.3f %9.3f %9.1f%% %9s" % (nm, mae, rng_j, frac * 100, verdict))

print()
print("总 MAE %.4f" % errs.mean())
print()
print("对照：训练结束时 l1_loss 约 0.045（归一化空间）。")
print("这里的数字如果和它同量级，说明模型在分布内是准的，问题在部署。")

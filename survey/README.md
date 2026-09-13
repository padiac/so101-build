# 生态调查 · 2026-09-03

别人在用什么。查的方法是 HuggingFace / GitHub 的 API 拿硬数据，不是凭印象。

一句话结论：**这个生态比我们以为的大得多，而且有好几样东西可以直接拿来用。**

---

## A. 能装进 10 GB 显存的 VLA（已下载）

按今天的教训，模型能不能用先看两件事：**权重多大**，以及**动作空间是不是关节角**。
Wall-OSS 就是栽在第二条（它输出末端笛卡尔位姿，SO-101 给不了）。

| 模型 | 权重 | 骨干 | 说明 |
|---|---|---|---|
| `lerobot/smolvla_base` | 0.91 GB | SmolVLM2-500M | **预训练数据就是 SO-100/101 社区遥操作**，动作空间同构。下载量 175k，生态第一 |
| `MINT-SJTU/Evo-1` | 3.10 GB | InternVL3-1B | 轻量 VLA，上交，`lerobot` 里有 `evo1` |
| `lerobot/xvla-base` | 3.52 GB | — | 另一个能装下的 |
| `lerobot/VLA-JEPA-Pretrain` | 6.16 GB | V-JEPA | JEPA 路线，和世界模型直接相关 |

装不下的：`pi05_base` 14.5 GB，`lingbot-vla-4b` 16.8 GB，`MolmoAct2-SO100_101` 21.8 GB。
这三个要微调都得上 Colab A100 级别。

## B. 现成的 SO-101 数据集（已下载三份）

HuggingFace 上搜 `so101` 返回 1000+ 个数据集。下载量前列：

- `lerobot/svla_so101_pickplace` — 官方参考
- `TommyZihao/lerobot_zihao_dataset_shake_hands` — 握手，LingBot 后训练用的就是它
- `TommyZihao/lerobot_zihao_dataset_a` — 夹放砂糖橘
- `hbseong/record-stacking-so101` — 叠放
- `armnet/armnetbench_v01_lerobot_so101` — 基准
- `jackvial/so101_pickplace_success_120_v2`

**价值不在数据本身，在于它们是别人趟过的任务定义和录制规格**，可以直接对照我们的做法。

## C. 已经有人在 SO-101 上做成的事

- **LingBot-VLA 后训练**（蚂蚁灵波）：握手任务，SO-ARM101 配置文件 + 权重 + 三个 UP 主的视频教程全公开。
  适配新机器人是写一个 `configs/robot_configs/*.yaml` 的键名/切片映射，不改模型。
- **`MINT-SJTU/Evo-RL`**：SO-101 上的**真机离线强化学习**，不是模仿学习。
- **`aalmuzairee/squint`**：SO-101 的 sim-to-real 视觉 RL。
- **`StoneT2000/lerobot-sim2real`**：仿真训练 → 零样本部署到真机。
- **`MuammerBay/isaac_so_arm101`**：SO-ARM101 的 Isaac Lab 工程。
- **`legalaspro/so101-ros-physical-ai`**：SO-101 的 ROS2 全栈。
- **`lehome-official/lehome-challenge`**：家庭任务比赛，`IliaLarchenko/lehome_solution` 是线上第一名的方案。

## D. 工具（已 clone 到 `repos/`）

| 仓库 | 解决什么 |
|---|---|
| `box2ai-robotics/lerobot-kinematics` | SO-100/101 的正逆运动学。下象棋那类"给坐标去位置"的项目直接要它 |
| `robocurve/inspect-robots` | **现成的策略评估框架**。我们一直缺的成功率自动化 |
| `villekuosmanen/physical-AI-interpretability` | 注意力可视化。看策略到底在看哪里 |
| `Tavish9/any4lerobot` | LeRobot 工具集 |
| `IliaLarchenko/dot_policy` | Decoder-only transformer 策略，ACT 之外的轻量选择 |
| `maximilienroberti/lerobotdepot` | 社区硬件索引 |
| `fracapuano/robot-learning-tutorial` | Robot Learning 教程全文 |

## E. 学习资源（中文为主，都没下载，是链接）

- `TianxingChen/Embodied-AI-Guide` — 15.8k ⭐，中文具身智能技术指南，生态里最大的一个
- `datawhalechina/every-embodied` — 3.5k ⭐，从 0 构建 VLA/OpenVLA/SmolVLA/Pi0
- `Xbotics-Embodied-AI-club/Xbotics-Embodied-Guide` — 学习路线 + 公司图谱
- `HCPLab-SYSU/Embodied_AI_Paper_List` — 论文列表
- 同济子豪兄飞书知识库 — SO-101 全流程中文教程，从买件到推理
- `knightnemo/Awesome-World-Models` / `leofan90/Awesome-World-Models` — 世界模型专门列表

## F. 值得知道但暂时用不上

- `OpenHelix-Team/VLA-Adapter` — 小规模 VLA 的范式
- `PRIME-RL/SimpleVLA-RL`、`RLinf/RLinf` — 用 RL 扩展 VLA 训练
- `RoboTwin-Platform/RoboTwin`、`mani-skill/ManiSkill`、`simpler-env/SimplerEnv` — 仿真与基准
- `dexmal/dexbotic` — VLA 工具箱
- `SwanHubX/SwanLab` — 训练可视化（wandb 的国产替代）

---

## 一个校准数字

GM-100 真机基准，三个平台平均：

| | SR | PS |
|---|---|---|
| WALL-OSS | 4.05% | 10.35% |
| GR00T N1.6 | 7.59% | 15.99% |
| π0.5 | 13.02% | 27.65% |
| LingBot-VLA | 17.30% | 35.41% |

同样这些模型在 RoboTwin 2.0 仿真里是 77–89%。

**真机和仿真差 5 倍，这是常态。** 看到 VLA 的漂亮数字，先问是哪一边。

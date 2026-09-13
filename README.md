# SO-101 装配日志

Hugging Face LeRobot **SO-ARM101** 双臂（leader + follower）搭建记录。
开始日期：2026-08-16

官方资料：
- 装配文档：https://huggingface.co/docs/lerobot/so101
- STL + BOM：https://github.com/TheRobotStudio/SO-ARM100
- 舵机规格：https://www.waveshare.com/st3215-servo.htm

---

## 目录结构

根目录只放**实际会跑的东西**，诊断脚本都在 `tools/` 下（见 `tools/README.md`）。

```
robot.ps1              唯一入口：ports / health / scan / teleop / record / eval / calibrate-*
cams.py                相机识别与映射（robot.ps1 每次自动调 resolve）
home.py                归位到该策略训练数据的起始姿态
health.py  scan_motors.py
record_win.py  rollout_win.py  teleop_win.py    打好补丁的 lerobot 入口包装
lerobot_patch.py       lerobot 的健壮性补丁（扭矩重试、exit_early 防护、检测框叠加）
patch_act_nostate.py   修 lerobot 在无本体感觉输入时的一个崩溃
train_act.sh           训练（可用环境变量覆盖 DATA/CHUNK/ASTEPS/USE_VAE/BATCH/LR/STEPS）
pipeline_v3.sh         拷贝 -> 裁剪 -> 训练，一条龙（给定时任务用）
trim_dataset.py        裁掉每条 episode 开头的静止段

tools/                 诊断脚本，按用途分四类，见 tools/README.md
datasets/              采集的数据集 + rollout_probe（评估录像）
policies/              训练好的 checkpoint
figures/               分析时生成的图，可随时重新生成
hardware/              改过的 3D 打印件（相机支架、控制板底板）
optional/              官方可选件的 STL/SCAD
logs/                  运行日志
deprecated/            已废弃但留作记录的脚本，附说明
```

**模型清单见 [MODELS.md](MODELS.md)** —— 每份数据当时在录什么、哪个能用、哪个不能用。
权威的参数表不要手抄，直接读 checkpoint：

```powershell
.\.venv-win\Scripts\python.exe tools\policy\list_models.py
```

一句话版：测指令跟随只能用 `smolvla_*`（**ACT 没有语言输入**，不管按哪个按钮都只做
它那份数据里的那一件事）；单任务最好的是 `act_v3rand_100k`；带 `_OLDCAMMAP` 的
是相机映射修正之前训的，**不能用**（会喂进对调的画面）。

---

## 我的配置

| | 舵机 | 减速比 | 堵转扭矩 | 电源 |
|---|---|---|---|---|
| **从臂 follower** | STS3215 **12V** ×6 | 全 1/345 | 30 kg·cm | **12V 8A**（官方要求 5A+，实配 8A 有余量）|
| **主臂 leader** | STS3215 **7.4V** ×6 | 混搭（见下） | — | **5V** |

电源插头规格：5.5×2.1mm，**中心正极**。板子 DC 输入直通舵机，可接受 6–12.6V，12V 在上限内。

官方原文：
> The 12V version has a stall torque of 30kg.cm. Note if you do this, you will also
> have to buy a 12V 5A+ power supply instead of a 5V one.
> **The leader arm is always 7.4V for the SO101.**

### ⚠️ 最高优先级警告

**两个电源电压不同，接口一模一样。主臂插上 12V = 6 个舵机当场烧毁。**

Waveshare Bus Servo Adapter (A) 的 DC 输入是**直通舵机**的，板子 6–12.6V 全都吃，
**不会报错、不会保护**。已在电源插头上贴标签区分。

- 12V → 从臂
- 5V → 主臂

---

## 主臂减速比对照（SO-101 特有）

主臂故意用低减速比，为了手动拖动时省力（SO-100 全 1/345 拖不动，是 101 修掉的问题）。

| 关节 | 电机 ID | 减速比 | 料号 |
|---|:---:|:---:|:---:|
| Shoulder Pan（底座旋转） | 1 | 1/191 | C044 |
| Shoulder Lift（大臂抬升） | 2 | 1/345 | C001 |
| Elbow Flex（肘） | 3 | 1/191 | C044 |
| Wrist Flex（腕俯仰） | 4 | 1/147 | C046 |
| Wrist Roll（腕旋转） | 5 | 1/147 | C046 |
| Gripper（夹爪） | 6 | 1/147 | C046 |

从臂 6 个全部 1/345（C001），ID 编号规则同上。

---

## 环境

- **LeRobot 跑在 WSL2 Ubuntu 24.04**，conda 环境名 `lerobot`（Python 3.10）
  - 理由：训练 ACT / Diffusion Policy 要吃 RTX 3080，GPU 直通已验证。
    Windows 原生跑 LeRobot 依赖链（pyav 等）是二等公民。
- **串口靠 usbipd-win 5.3.0 转发**进 WSL
  - WSL 内核 5.15.146.1，`CONFIG_USB_ACM=y`（内置），驱动板转发过去直接是 `/dev/ttyACM0`，无需重编内核
  - 转发脚本：[`attach-usb.ps1`](attach-usb.ps1) —— **每次拔插 USB 后都要重跑**
- 相机：Logitech Brio 100（`046d:094c`）已在机

---

## 进度

- [x] 3D 打印件（买时已备齐）
- [x] 舵机、电源、驱动板、线材（买时已备齐）
- [x] 安装 usbipd-win 5.3.0
- [x] 写 `attach-usb.ps1` 转发脚本（+ `watch-usb.ps1` 插拔诊断脚本）
- [x] 串口链路打通：Windows → usbipd → WSL vhci_hcd → cdc_acm → `/dev/ttyACM0`
- [x] 安装 LeRobot（conda env `lerobot`）
      - `lerobot 0.4.4` / `torch 2.10.0+cu128` / `feetech-servo-sdk 1.0.0` / `pyserial 3.5`
      - `torch.cuda.is_available() == True` → NVIDIA GeForce RTX 3080 ✅
- [x] **从臂**设舵机 ID / 波特率（6 个全部完成）
- [x] **从臂**总线扫描验收通过：波特率 1000000，ID 1–6 齐全，型号码全部 777 (STS3215)
- [x] **主臂**设舵机 ID / 波特率（6 颗完成，减速比对号入座）
- [x] **主臂**总线扫描验收通过
- [x] 清支撑
- [x] 从臂装配 Joint 1→5 + Gripper  **（2026-08-17 完成）**
- [x] **主臂**装配 Joint 1→5 + Handle/Trigger  **（2026-08-19 完成）**
- [x] 标定 calibrate（两臂完成，行程 span 主从差 <5%，drive_mode 全 0）
- [x] 遥操作跑通  **（2026-08-19）**
- [x] **硬件工作全面迁到 Windows**（`robot.ps1` 统一入口，2026-08-22）
- [x] Windows 遥操作稳定跑通：33.34ms / 30Hz，无掉线无过载
- [x] 相机装好 + 调焦（顶部 index 1 / 手腕 index 0，清晰度 824→1579）
- [x] 相机索引锁定 + 内容校验（`cams.py`，分离度 47 倍）
- [ ] 采集示教数据集 ← 当前步骤
- [ ] 训练策略

---

## 关键命令

```bash
# 激活环境
conda activate lerobot

# 找串口
lerobot-find-port

# 设舵机 ID（装配前做！脚本从 gripper 倒着问到 shoulder_pan，一次只接一个电机）
lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/ttyACM0
lerobot-setup-motors --teleop.type=so101_leader   --teleop.port=/dev/ttyACM1

# 标定
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower
lerobot-calibrate --teleop.type=so101_leader   --teleop.port=/dev/ttyACM1 --teleop.id=my_leader

# 串口权限
sudo chmod 666 /dev/ttyACM0
```

---

## 踩坑记录

### 0. ⭐ 板子完全不被识别 —— 用了充电线，不是数据线（已解决）

**症状**：驱动板接好 12V 电源 + USB，`usbipd list` 里**一个新条目都没有**，
设备管理器里也没有任何未知设备。

**诊断要点**：如果只是缺驱动，设备**照样会出现**在列表里（显示成
`Unknown USB Device` 或带黄色感叹号）。**连"有东西插上了"都检测不到 = 物理层问题，
不是驱动问题。** 这一条能立刻把排查范围从软件切到硬件。

**原因**：用了一根**只有电源线、没有数据线**的 USB-C 充电线。外观和数据线完全一样。

**验证方法**：拿同一根线接手机到电脑。手机只充电、电脑里不出现设备 = 线是废的。

> 换数据线后立刻枚举成功。以后板子"没反应"，**第一个怀疑对象永远是线。**

### 1. WSL 看不到串口（已解决）
WSL2 默认不直通 USB。`lerobot-find-port` 在 WSL 里啥也找不到。
解法：usbipd-win。内核已内置 `USB_ACM=y`，转发后直接出 `/dev/ttyACM0`。

### 2. 驱动板信息（已确认可用）
- 芯片：**WCH CH343**，`VID:PID = 1a86:55d3`，序列号 `5B3E090575`
- Windows：内置 usbser 驱动直接认成 `USB Serial Device (COM8)`，**不需要装 CH34x 驱动**
- WSL：`cdc_acm` 绑定为 `/dev/ttyACM0`，属主 `root:dialout`
- `padiac` 已在 `dialout` 组，可直接打开，**不需要每次 chmod 666**
- LeRobot 环境里 pyserial 确认可见

### 3. attach-usb.ps1 误报 "none found"（已修）
attach 返回后 WSL 侧枚举还要 1–2 秒，脚本立刻 `ls` 会扑空，误导成失败。
已改成轮询最多 10 秒。**dmesg 才是判断依据**：
```
usb 1-1: New USB device found, idVendor=1a86, idProduct=55d3
cdc_acm 1-1:1.0: ttyACM0: USB ACM device
```

### 4. mirrored 网络模式不影响 usbipd（已验证）
attach 时会打印 `Detected networking mode 'mirrored'. Using IP address 127.0.0.1`。
一度怀疑撞上本机 mirrored loopback 的老问题，实测没有：Windows 侧 3240 端口
`Listen` + `127.0.0.1 Established`，链路正常。

### 5. 关于 `vhci_hcd NOT loaded` 的误判
`lsmod | grep vhci` 查不到不代表没有。WSL 内核 `CONFIG_USBIP_VHCI_HCD=y` 是
**编译进内核**的，本来就不出现在 `lsmod` 里。判断依据看 dmesg。

### 7. 舵机盘装反顺序，拔不下来（Joint 5）

**症状**：Joint 5 按前四个关节的习惯先装了盘，结果电机塞不进 wrist holder。
盘是过盈配合，装上去拔不下来。

**拔盘方法（不要撬！）**：撬棍别在舵机壳体上，力全压在输出轴承上，会把轴撬旷。

**✅ 用盘上那 4 个 M3 孔做顶丝**（自制拔轮器）—— **2026-08-16 实战验证有效，一次成功**：

> 原理：把"撬"变成"顶"。受力从别扭的侧向变成纯轴向，且靠螺纹慢慢加力、力度可控。
> 通用心法：**遇到过盈配合要拆，先找有没有地方能做顶丝，基本都比撬强。**

1. 准备 2 颗**更长的** M3 螺丝（M3×16 或 M3×20）
2. 拧进盘上**对角**的两个 M3 孔
3. 继续拧，螺丝顶到舵机壳体端面
4. **两颗交替拧，每次半圈** —— 单边猛拧会顶歪卡死

配合**吹风机热风吹盘 30–60 秒**（POM 受热膨胀软化）。别用热风枪，
别对着壳体长吹（里面有磁编码器）。

实在不行买小型两爪拔轮器（二三十块）。

### 8. 螺丝卡在 3D 打印件里取不出来

打印孔偏小（热收缩 + 象脚效应）+ 螺丝自攻切螺纹，摩擦力很大。

- **🔥 打印件专属大招**：PLA 玻璃化温度只有 ~60°C。**用电烙铁尖碰螺丝头 3–5 秒**，
  周围塑料软化，一拧就出来。金属件上没这招
- **螺丝刀尺寸必须对**：M3 十字用 **PH1**，M2 用 **PH0**。刀头小一号就打滑(cam-out)，
  一打滑十字槽就圆了。八成的"螺丝拧坏"其实是"螺丝刀选小了"
- **七分压三分转**：往下压的力要远大于旋转的力
- 螺丝头拧圆了：橡皮筋垫在刀头和螺丝头之间；或用断丝取出器
- 脱扣但卡在深槽里取不出：缝衣针弯个小钩（回形针太粗）／从背面顶／气吹／
  蓝丁胶或黄油粘在刀头上／磁化螺丝刀（要用钕磁铁，单向蹭 20–30 下）

**长期方案**：反复拆装的位置上**热熔铜螺母**，塑料自攻螺纹拧三五次就滑丝。

### 9. 装配前忘了核对电机 ID（Joint 1 装错）

**教训**：设完 ID 立刻**贴标签**，别指望记住。

**好消息**：从臂 6 颗物理上完全一样（都 12V / 1/345 / 型号码 777），ID 只是
EEPROM 里的软件标签。而且 SO-101 改进了走线，**装配后 3-pin 接口仍然够得着**，
所以装错了不用拆 —— 单独连上那颗重写 ID 即可（`setup_motor` 是覆盖写，幂等）：
```bash
python setup_motors.py follower shoulder_pan   # 只连那一颗
```
⚠️ 主臂不适用 —— 主臂减速比不同，装错位置必须真拆。

### 10. 控制板装不上 —— 官方安装板的孔和 M2.5 铜柱零间隙（已自制改件）

**症状**：`WaveShare_Mounting_Plate_SO101` 上四个孔是**水滴形**，铜柱怎么都怼不进去。
另一块 `Seeedstudio_Mounting_Plate_SO101`（M2 沉头孔，好装）**孔距又不对**——
那是给 Seeed 的板子设计的，不是 Waveshare 的。

**官方完全没有这部分教程**：装配文档的视频只到 Joint 1–5 + Gripper，
控制板怎么装一个字没提，网上也搜不到有人记录这个问题。

**实测原始件几何**（用射线投射切片量的，见 `scratchpad/slice.py` 思路）：
```
平板       51.0 x 42.0 x 4.0 mm （中间凸台到 z=7.6，是底座卡扣）
孔心       (±18.5, ±14.0)  ->  孔距 37.0 x 28.0
孔型       水滴：Ø5.0 圆 + 1.0mm 尖顶（尖朝 -Y）
```

**根因**：孔是 **Ø5.0**，而 **M2.5 六角铜柱对边正好 5.0mm** —— 零间隙，冷压进不去。

> 顺带：水滴形不是功能设计，是 **FDM 打印惯例**——横向圆孔顶部悬空会塌，
> 改成尖顶把悬垂角控制在 45° 内就不用支撑。整条臂上到处都是水滴孔，都是这个原因。

**解法**：自制改件 —— 填掉四个水滴孔，重开 **M2.5 直壁沉孔（counterbore）**。

> ⚠️ **第一版做错了**：开成了锥形沉头（countersink）。锥孔只配 DIN 965 那种
> **锥面平头**螺丝，而**铜柱套装配的从来都是圆柱头**，圆柱头放进锥孔必然凸出来。
> **直壁沉孔才对**。错的文件已删。
>
> 记住区别：**countersink = 锥面（配平头）／counterbore = 直壁平底（配圆柱头）**。

| 文件 | 说明 |
|---|---|
| `WS_Plate_M25_counterbore_2.6mm_socketcap.stl` | **默认打这个**，沉孔深 2.6mm，通吃内六角/盘头 |
| `WS_Plate_M25_counterbore_2.0mm_panhead.stl` | 沉孔深 2.0mm，只用盘头/圆柱头，留更多板厚 |
| `WS_Plate_M25_counterbore.scad` | 源文件，参数具名可改 |
| `ws_plate_ORIGINAL.stl` | 官方原件，留作对照 |

**装配方式**：
```
板子 + 铜柱  ->  平的那面 (z=0)
机械臂底座   ->  凸台那面 (z=4)
M2.5 螺丝从凸台面穿下去锁进铜柱，螺丝头整个坐进沉孔里
```

**螺丝头高度参考**（选 `CB_DEPTH ≥ 头高`）：

| 类型 | 头径 × 头高 |
|---|---|
| M2.5 内六角 DIN 912 | 4.5 × **2.5** |
| M2.5 十字盘头 DIN 7985 | 5.0 × **1.8** |
| M2.5 圆柱头 DIN 84 | 4.5 × **1.6** |

长度买 **8mm** 左右（穿 4mm 板 + 咬进铜柱）。

**改件参数**：
- `M25_CLEAR = 2.8` 通孔径
- `CB_DIA = 5.4` 沉孔径 / `CB_DEPTH` 沉孔深（2.6 或 2.0）

⚠️ **沉孔开在凸台面是我的判断**（基于"底座接触面不能有凸出"）。如果装的时候发现
该开在平的那面，把 `.scad` 里 `m25_counterbore()` 的沉孔那行改成：
```openscad
translate([0, 0, -1]) cylinder(d = CB_DIA, h = CB_DEPTH + 1);
```
然后重新导出。

**验证**（切片实测两个导出件）：z=1.0 → Ø2.800 正圆通孔；z=3.5 → Ø5.400 直壁。

### 11. 只装一个舵机盘的两处（容易以为装错了）

前四个关节都是**两个盘**（双端支撑，扛 30kg·cm 力矩，防止输出轴被弄旷）。
但有**两处只装一个**，官方文档用 both / a 刻意区分了：

| 位置 | 盘数 | 官方原文 |
|---|:--:|---|
| Joint 5 wrist_roll | **1** | "Install **only one** motor horn on the wrist motor" |
| 主臂 Handle/Trigger | **1** | "attach **a** motor horn using a M3x6mm horn screw" |
| 从臂 Gripper | 2 | "Install **both** motor horns on the gripper motor" |

**为什么扳机只要一个**：扳机不承力，受力只有手指扣的那点力，是个绕输出轴转的
杠杆，单端驱动就够。另一侧根本没有对应结构件，盘装上去也固定不住 —— 如果你
发现"这个盘装上没意义也固定不住"，那就是对的，别硬装。

**剩料自检**（每颗舵机配 2 个盘，一条臂 6 颗 = 12 个）：

| 臂 | 用掉 | **应剩** |
|---|:--:|:--:|
| 主臂 | Joint1-4 用 8 + Joint5 用 1 + Trigger 用 1 = 10 | **2** |
| 从臂 | Joint1-4 用 8 + Joint5 用 1 + Gripper 用 2 = 11 | **1** |

装完数一下剩几个，对得上就说明没漏没多。

### 12. 僵尸 attach —— usbipd 显示 Attached 但 WSL 里没设备

**症状**：`usbipd list` 显示 `Attached`，但 WSL 里 `ls /dev/ttyACM*` 什么都没有，
`lerobot-calibrate` 报 `could not open port /dev/ttyACM1: No such file or directory`。

**原因**：物理拔插 USB 之后，Windows 侧的 attach 记录**不会自动清理**。
状态卡在 "Attached" 上，此时再 attach 一次会**静默无效**（它认为已经接上了）。

**解法**：先 detach 再 attach。已加进 `attach-usb.ps1`，现在跑一次就自愈：
```bash
powershell -ExecutionPolicy Bypass -File E:\Repo\so101-buildttach-usb.ps1
```

手动版：
```
usbipd detach --busid 3-3
usbipd attach --wsl --busid 3-3
```

> **判断口诀**：`usbipd list` 说 Attached ≠ WSL 里真有设备。
> **以 WSL 侧的 `ls /dev/ttyACM*` 为准。**

### 13. 双板端口对照

两块驱动板同型号（都是 `1a86:55d3`），靠**序列号**区分：

| WSL 设备 | 序列号 | 板子 | 臂 |
|---|---|---|---|
| `/dev/ttyACM0` | `5B61033038` | CH343 (COM9) | 主臂 leader |
| `/dev/ttyACM1` | `5B3E090575` | USB Serial (COM8) | 从臂 follower |

⚠️ **ttyACM 编号不保证稳定**——取决于 attach 顺序。不确定时用序列号核对：
```bash
python -c "from serial.tools import list_ports; [print(p.device, p.serial_number) for p in list_ports.comports()]"
```

### 14. ⭐ shoulder_lift 反复 Overload 跳闸 —— 标定下限压在机械死点上

**症状**：遥操作一启动，从臂 shoulder_lift(ID 2) 立刻跳闸，LED 闪烁、关节不动，
退出时崩在 `Failed to write 'Torque_Enable' on id_=2 ... [RxPacketError] Overload error!`

**日志特征**（关键判据）：
```
'shoulder_lift': {'original goal_pos': -92.4, 'safe goal_pos': -93.615}
'shoulder_lift': {'original goal_pos': -68.0, 'safe goal_pos': -93.615}   <- 不变
```
`safe = present + clamp(goal-present)`。**safe 死钉不动 = present 不动 = 关节已堵转**。
后面几千行 warning 都是同一个症状的回声，不是新问题。

**排查过程（每步只排除一个变量）**：

| 步骤 | 结果 | 排除掉 |
|---|---|---|
| 断电用手掰关节 | 非常灵活、无卡滞 | 机械干涉 |
| 读保护寄存器 `protect.py` | 全是出厂默认，和其他关节一致 | 阈值被调低 |
| 单关节抬升测试 `lift_test.py` | -105°→-68° 峰值负载 **164/1000 = 16%** | 舵机能力 / 供电 |
| 对比 range_min 和静止位置 | **找到原因** | |

**根因**：标定"摇行程"时是**用手把关节推到机械死点**才记录 min 的：

| | range_min | 断电静止 raw | 差 |
|---|---|---|---|
| 主臂 shoulder_lift | 925 | 939 | 14 |
| **从臂 shoulder_lift** | **841** | **890** | **49 (~4.3°)** |

遥操作把主臂的极限映射成从臂的极限 → 从臂被指令到 841 → **往下顶死点** →
堵转 → 持续满扭矩 → 触发过载保护。而且**每次启动都会发生**，因为断电后
两条臂都在重力下塌到底部。

**解法**：`shrink_range.py` 给标定范围上下各留 60 counts 余量：
```bash
python shrink_range.py --margin 60          # 预览
python shrink_range.py --margin 60 --apply  # 写入（自动备份）
```
从臂 shoulder_lift 下限 841 → **901**，比静止位置 890 还高 11 counts，
指令永远够不到死点。

改完启动遥操作会提示 `Press ENTER to use provided calibration file` ——
**直接回车**（按 `c` 会重新标定，前功尽弃）。

**验证**：改后 `safe goal_pos` 42.16 → 43.04 → 45.86 → 49.73 一路收敛到目标，
warning 随即消失，进入正常跟随。

> **教训**：不要一上来就调高保护阈值。那样跳闸是压住了，但舵机会真的持续堵转
> 发热，问题被掩盖成更坏的形式。**先排除变量，再动参数。**

### 15. 诊断工具箱

本目录下的脚本，按用途：

| 脚本 | 用途 |
|---|---|
| `attach-usb.ps1` | Windows→WSL 串口转发（含 detach 自愈） |
| `watch-usb.ps1` | USB 插拔事件监视，判断物理层通不通 |
| `setup_motors.py` | 设舵机 ID，默认 ID 正序 1→6，带重试 |
| `scan_motors.py` | 总线扫描验收，确认 ID 1-6 齐全 |
| `diag_motors.py` | 位置/Homing_Offset，`--watch` 实时刷新 |
| `watch_both.py` | 双臂对照，看主从方向是否一致 |
| `health.py` | 错误标志/电压/温度/负载 |
| `protect.py` | 过载保护寄存器，`--boost` 可调阈值 |
| `lift_test.py` | 单关节受控抬升 + 负载曲线 |
| `teleop_diag.py` | 带负载记录的遥操作，超限自动停机 |
| `shrink_range.py` | 给标定范围加安全余量 |
| `run_teleop.sh` | 带日志的遥操作（前后各抓一次 health） |

### 16. 可选改装件（官方 Optional 目录，已下载到 `optional/`）

官方仓库 `Optional/` 下有一堆没写进主装配文档的改装件。已下载的：

#### 柔顺夹爪（解决"夹不稳 / 夹太紧"）
`optional/gripper/` —— 内部掏空 + 加强筋，抓取时顺着物体变形。

| 项 | 值 |
|---|---|
| 材料 | **TPU 95A** |
| 填充 | 20%，**要支撑**（斜口钳剪） |
| 额外五金 | **无** —— 外形与原件一致，直接替换 |

⚠️ 需要打印机能打 TPU（近程直驱友好，远程 Bowden 易堵）。
打不了就退回**贴 3M 防滑胶带**，官方也提到这个选项。

> 为什么比贴垫子好：这是**结构性柔顺**，整个爪在变形，接触面积大、
> 受力均匀，不会像贴片那样蹭掉或滑移。同时降低所需握力、增加容错。

#### 手腕相机支架
`optional/wristcam/` —— **Brio 100 装不上手腕**（太大太重），需另买
**32×32mm USB 相机模组（≥720p/30fps）**，二三十块。

- `SO-ARM101_camera_wrist_mount.stl` —— **推荐**，六角螺母沉槽版
- `Wrist_Cam_Mount_32x32_UVC_Module_SO101.stl` —— 旧版

五金：4× M2（舵机套件自带）+ 2× M3×8 + 2× M3 六角螺母
打印：**填充 40%**（官方强调，防晃），树形支撑
装法：螺母嵌进 Wrist Roll Follower 沉槽 → 装回 6 号电机 → 2 颗 M3
⚠️ 模组是**手动对焦**，通电后拧镜头调清晰

#### 顶部相机支架（Brio 100 用这个）
`optional/overheadcam/` —— `arm_base` + `cam_mount_bottom` + `cam_mount_top`

#### 官方还有但没下的
`Wrist_Cam_Mount_RealSense_D405 / D435`、`Overhead_Cam_Mount_32x32_UVC_Module`、
`Wrist_Cam_Plug_Mount_32x32_UVC_Module`

> 这些 Optional 件**没有官方视频**，只有 README + `media/` 目录里的实物照片。

### 17. 相机方案

#### ⚠️ WSL 不支持 USB 摄像头（内核层面）
```
内核 5.15.146.1-microsoft-standard-WSL2
config 总行数 1492（极简内核）
MEDIA / V4L2 / UVC 相关配置：0 条
uvcvideo、videodev 模块：不存在
```
usbipd 能把摄像头转发进去，但**没有驱动认它**，不会出现 `/dev/video*`。
串口能用是运气好（`CONFIG_USB_ACM=y` 内置），摄像头这边是零。

**方案 B（采用）**：Windows 采数据 → WSL 训练。
- Windows 原生认摄像头（DSHOW），串口也直接是 COM 口，**不用 usbipd**
- 数据集是文件，放 `E:\` 两边都能读；训练仍在 WSL 吃 GPU
- Windows venv: `.venv-win/`（Python 3.13）

⚠️ **版本必须对齐**：`lerobot 0.6.1` 要求 Python ≥3.12，`0.4.4` 要求 ≥3.10。
WSL 是 py3.10 → 装到 0.4.4；Windows 是 py3.13 → 装到 0.6.1。
**两边不一致，数据集格式可能对不上。** 计划把 Windows 钉到 0.4.4 对齐。

方案 A（备选）：重编 WSL 内核开 `CONFIG_USB_VIDEO_CLASS`
（[microsoft/WSL2-Linux-Kernel](https://github.com/microsoft/WSL2-Linux-Kernel)）。

#### Brio 100 实测（Windows）
`640x480 / 实测 30.7fps / YUY2 / DSHOW 后端` —— 可用。

⚠️ **两个坑**：
1. **隐私挡片**要打开（实测踩过，画面全黑）
2. **必须预热**：刚打开的前几十帧是黑的，自动曝光没收敛。
   `cam_test.py` 已加 20 帧预热 + 亮度检测

#### 相机选型：什么影响性能（按重要性）
1. **视角布置** —— 全局相机给上下文，**手腕相机给精细操作**（视野相对夹爪，最关键）
2. **一致性** —— 关自动对焦、锁曝光、位置固定。免费但比换贵相机管用
3. **延迟** —— 画面和动作在时间上对不齐，策略会学到滞后的映射。设 `CAP_PROP_BUFFERSIZE=1`
4. **帧率稳定**
5. **视场角** —— 手腕相机要 90–120°，太窄物体会出画
6. **分辨率** —— **最不重要**，策略缩到 ~224×224，640×480 足够
7. **全局快门** —— 手腕位置加分项，非必需
8. **MJPEG** —— 2 个以上相机才要紧。YUY2 640×480×30 约 18MB/s，
   USB2.0 实际约 35MB/s，两个 YUY2 就撑满一个控制器

#### 结论：两个视角都用 32×32 UVC 模组
**Brio 100 装不上官方顶部支架** —— 官方要求"拆掉原装底座"用 M2 固定，
而 Brio 100 **没有 1/4-20 螺纹，只有不可拆的显示器夹**。
官方支架是给特定 webcam（Amazon B082X91MPP）设计的。

改用同款 32×32 模组 ×2：成像特性一致、官方支架都现成、便宜、轻、好买定焦版。
Brio 100 留着开会用。

已下载到 `optional/`：`wristcam/`、`overheadcam/`（webcam 版）、
`overheadcam_32x32/`（推荐，多一个 `cam_mount_middle` 加高件）

⚠️ **相机位置一旦固定就不能再动** —— 策略学的是"这个视角的画面→动作"，
中途挪相机会让之前的数据全部作废。采数据前拧死。

### 18. 手腕相机支架开口太小（已自制改件）

**症状**：U20CAM-1080P 装不进官方 `SO-ARM101_camera_wrist_mount`。
4 个螺丝孔对得上，但**中央开口太小**，相机怎么转都塞不进去。
板子太厚，手工削不现实。

**实测原件几何**（射线投射切片；这块板是**斜的**，不与任何主轴平行，
先用法向聚类找出板平面，转正后再切）：
```
板法向     n = (0.423, 0.906, 0)      # XY 平面内转了 65 度
板厚       4.2 mm
4 个安装孔  (-42.10,4.00) (-42.10,31.00) (-15.10,4.00) (-15.10,31.00)
           -> 孔距 27.0 x 27.0 mm，孔径约 2.2
原开口     外接框 16 x 25，中心 (-28.53,17.55)，面积 310mm2（十字形）
板轮廓     u -45.03..-8.63   v 0.20..35.00
```

**为什么用十字形而不是方孔**：四个角被螺丝孔卡死，但上下左右的**中段**
还有余量。十字正好吃掉这些余量，在保证孔周围有筋的前提下把开口做到最大。

**结果**：

| | 原件 | 改后 |
|---|---|---|
| 开口外接框 | 16.00 x 25.00 | **28.00 x 29.00** |
| 开口面积 | 310 mm2 | **731 mm2**（2.4 倍） |
| 最小通过宽度 | 16.0 | **19.5** |
| 4 个螺丝孔 | — | 位置尺寸**未变** |
| 孔周围最小筋宽 | — | ~2.7 mm |

**文件**（`optional/wristcam/`）：
- `SO-ARM101_camera_wrist_mount_BIGGER.stl` —— 打这个
- `wrist_cam_mount_bigger.scad` —— 源文件，`VBAR_W/VBAR_H/HBAR_W/HBAR_H` 四个参数可调
- `SO-ARM101_camera_wrist_mount.stl` —— 官方原件，对照用

#### 顶部相机支架同样要改

`overheadcam_32x32/cam_mount_top.stl` 是同一个模组的支架，**同样的病**。

| | 板法向 | 板厚 | 孔距 | 原开口 |
|---|---|---|---|---|
| 手腕件 | (0.423, **+0.906**, 0) | 4.2 mm | 27×27 | 16.0 × 25.0 |
| 顶部件 | (0.423, **−0.906**, 0) | 3.0 mm | 27×27 | 15.2 × 26.8 |

两者是 **Y 镜像**关系，孔位换算到开口中心都是 (±13.5, ±13.5)，
**所以同一套开口参数直接复用**，改后都是 28 × 29 / 731mm²。
两个视角的相机装上去成像位置一致，也算意外的好处。

切之前确认过：**开口区域法向上只有板本身，立柱不在后面**，可以放心切穿。
改后板轮廓（u跨 63.20 / v跨 36.60）与原件一致，立柱未受影响。

文件（`optional/overheadcam_32x32/`）：
- `cam_mount_top_BIGGER.stl` —— 打这个
- `cam_mount_top_bigger.scad` —— 源文件
- `cam_mount_top.stl` —— 官方原件

**19.5mm 已接近结构上限**（受 27mm 孔距限制）。还不够的话只有两条路：
把筋减到 ~1.5mm 换 ~21mm，或者放弃 4 点固定改 2 点，开口能大很多。

**相机手册没有机械图** —— 只给了 32x32mm、4 个 Ø2.2 安装孔、镜头座间距 18mm。
接插件位置只能实测。PDF 有加密标记，`pypdf` 重写一遍即可提取文本。

### 19. 相机实操踩坑

#### opencv-python-headless 没有 GUI
LeRobot 依赖的是 **headless 版**，`cv2.imshow` 会直接报
`The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support`。

Windows venv 里已换成完整版：
```bash
pip uninstall -y opencv-python-headless
pip install opencv-python          # 装到了 cv2 5.0.0
```
LeRobot 照常工作（它只要 `cv2`，完整版是超集）。`pip check` 会报
`lerobot requires opencv-python-headless, which is not installed` —— **名义警告，可无视**，
但别让别的东西把 headless 装回来。

`focus_cam.py` 也做了降级处理：GUI 不可用时自动切纯数字模式，不崩。

#### 相机全黑的两个原因
1. **隐私挡片**（Brio 100 有）—— 实测踩过两次
2. **没预热** —— 刚打开的前几十帧是黑的，自动曝光还没收敛。
   `cam_test.py` / `focus_cam.py` 都做了 20 帧预热

#### M12 镜头必须手动调焦
出厂焦距随机，U20CAM-1080P 拿到手是虚的。用 `focus_cam.py`：
拉普拉斯方差作清晰度指标（中央区权重 0.6），边拧边看数字。

- 对准**有细节的物体**，别对白墙（白墙方差恒为 0，仪表失效）
- 距离用**实际工作距离**（手腕 10–20cm / 顶部 30–50cm）
- ⚠️ **调好立刻用指甲油或热熔胶点在镜头螺纹接缝上固定**，
  否则焦点会慢慢漂，而且不会立刻发现 —— 等发现时数据已经采了一堆

#### ⚠️ OpenCV 相机 index 不稳定
插拔顺序、开机顺序变了，`index 0` / `index 1` 可能对调。
**顶部和手腕视角互换 = 策略学到的全错，而且不一定看得出来。**
位置固定后用 `lerobot-find-cameras` 拿稳定标识符，写死进配置再采数据。

### 20. ⭐ 串口问题反复出现的真正原因：存在两条路

**症状**：每次开工都要处理接口问题 —— attach 掉了、僵尸 attach、
`/dev/ttyACMn` 编号变了、跑错脚本、高负载下 usbipd 断链。

**根因不是某一次配置错，是「WSL 和 Windows 两条路并存」这件事本身。**
每次都要先想"板子现在在哪边"，而这个状态会被任何一次拔插/重启/测试改变。

**而且 WSL 这条路本来就走不通**：内核零 UVC 支持，相机永远用不了，
而采数据必须相机和机械臂在同一个进程里。

**永久解法：砍掉一条路。**

```
Windows  ->  所有硬件（遥操作、标定、扫描、相机、采数据）
WSL      ->  只训练，吃 3080
数据集   ->  放 E:\，两边共用
```

`robot.ps1` 是唯一入口，自愈设计：
- **自动把板子从 WSL 抢回来**（检测到 Attached 就 detach）
- **按序列号解析 COM 口**，COM 编号变了也不影响
- 自动加载 `lerobot_patch`（enable_torque 重试）
- 子命令：`ports` / `health` / `scan` / `teleop` / `calibrate-follower` / `calibrate-leader`

旧的 `run_teleop.sh` 改成打印提示并退出；原文件存为 `deprecated_*`。

**实测结果**（2026-08-22）：`Teleop loop time: 33.34ms (30 Hz)` 稳定，
三类老故障（usbipd 断链 / 上电丢包 / 僵尸 attach）全部消失。

#### 附带修掉的：PowerShell 把 stderr 当错误
lerobot 的日志走 stderr，PowerShell 5.1 会把每行包成 ErrorRecord 并打
`NativeCommandError` 横幅，把真正的输出淹掉。`robot.ps1` 里用
`Invoke-Native` 把 stderr 拍平成字符串。

### 21. 两个相机无法区分 —— 用画面内容校验

**问题**：两个 U20CAM-1080P **没有序列号**：
```
USB\VID_0C45&PID_6366&MI_00&29CD3151&0&0000
USB\VID_0C45&PID_6366&MI_00&183AF011&0&0000
```
后半段是 **USB 端口路径**派生的实例 ID，不是序列号。VID:PID 相同、
DirectShow 名字相同。OpenCV / pygrabber / `lerobot-find-cameras` 都只给
索引号 —— **在 Windows 上没有任何办法从设备层面区分它们**。

而索引顺序会随插拔和开机顺序变。

**为什么这个故障特别危险**：顶部和手腕视角互换后，训练**照常跑完、loss 照常下降**，
但策略学到的是错的映射。要到部署失败才发现，而那时数据已经采了几十个 episode。
这是典型的**静默损坏**。

**解法**：`cams.py` —— 存参考图，用画面内容比对。
```bash
python cams.py list                    # 抓图存联系表，肉眼确认
python cams.py set --top 1 --wrist 0   # 记映射 + 存参考图
python cams.py verify                  # 每次采数据前跑
```

实测分离度：对自己 1.8 / 对另一个 84.5，**47 倍**。两个视角差异极大，判定极可靠。

**当前映射**：`index 0 = 手腕`，`index 1 = 顶部`

#### 相机调焦
M12 镜头出厂焦距随机。`focus_cam.py` 用拉普拉斯方差做指标，
实测顶部相机 **824 → 1579**（翻倍）。调完用指甲油固定螺纹。

### 6. PowerShell 脚本编码
Windows PowerShell 5.1 按 **ANSI(GBK)** 解码 `.ps1`（除非有 UTF-8 BOM）。
脚本里写中文注释会乱码并连带撞出 `Missing closing ')'` 这类假语法错误。
**本目录所有 .ps1 保持纯 ASCII。**

### 22. 录制时按键没反应 —— `pynput` 没装，退回了终端监听器

**现象**：录制中按 `→` / `n` 结束当前 episode 没有任何反应，5 条全部录满
`EpisodeTime`（600 帧 = 20.0 秒，一条不差）。换蓝牙键盘也一样没用。

**根因**：lerobot 的 `init_keyboard_listener()` 有两个后端，按能力自动选：

| 后端 | 条件 | 行为 |
|---|---|---|
| `pynput` 全局监听 | `pynput_can_capture()` 为真 | **系统级抓键，焦点在哪都收得到** |
| `TerminalKeyListener` | pynput 不可用且 stdin 是 TTY | 只读当前终端的 TTY，**必须终端聚焦** |
| 无 | 都不满足 | 只能靠超时和 Ctrl+C |

`lerobot[viz]` 等 extra 都不带 `pynput`，所以默认落到第二档。而录制时人的
眼睛和手都在机械臂上，终端根本不是焦点窗口 —— 于是"按了没反应"。
换键盘完全无济于事，因为瓶颈不在键盘，在**哪个窗口有焦点**。

**修复**：

```bash
pip install pynput
```

装完验证（这一步要单独做，别混在录制里）：

```bash
python -c "from lerobot.utils.keyboard_input import pynput_can_capture; print(pynput_can_capture())"
# True = 已切到全局监听
```

**单独测输入层**：`keytest.py` 只起监听器、不碰串口和相机，跑起来后
**故意把焦点切到别的窗口**再按键。收得到 = 全局监听生效。

> **教训**：录制跑起来牵扯串口、相机、Rerun 一大堆东西，按键没反应时
> 分不清是哪一层。**把可疑的那一层单独拎出来测**，比在完整链路里猜快得多。
> 这和第 20 条是同一个道理 —— 先消掉变量，再动参数。

**顺带**：`play_sounds` 默认 `True`，Windows 下走 `System.Speech` 语音播报
"Recording episode 0" / "Reset the environment"。**手忙的时候靠耳朵判断阶段**，
比盯 Rerun 窗口靠谱得多，记得开音量。

**按键对照**（两个后端都支持）：

| 键 | 作用 |
|---|---|
| `n` / `→` | 结束当前阶段，进下一个（录制和重置阶段都能提前跳） |
| `r` / `←` | 重录这条 |
| `q` / `Esc` | 停止录制 |

`n` / `r` / `q` 是单字节，比方向键的转义序列可靠 —— 后者在 Windows 终端和
高延迟 SSH 下会被拆分、延迟或拦截。

**副作用**：按键能提前结束后，`-EpisodeTime` / `-ResetTime` 的含义从"时长"
变成"上限"，可以放宽（30 / 20），不用再卡秒去凑动作节奏。

---

### 23. 录到一半崩溃：`add_frame` 空 episode —— 全局监听把按键锁存了

**现象**：录到第 10 条时进程崩掉，已录的 9 条完好。

```
ValueError: You must add one or several frames with `add_frame`
            before calling `add_episode`.
  at lerobot_record.py:517  dataset.save_episode()
```

**根因**：`record_loop` 把按键检查放在循环最顶端、**加帧之前**：

```python
while timestamp < control_time_s:
    if events["exit_early"]:
        events["exit_early"] = False
        break            # <- 一帧都没录就退出
```

而 `record()` 主循环里 `save_episode()` 要编码 mp4，**耗时数秒**
（日志里那一堆 `Starting second pass: moving the moov atom` 就是它）。
这几秒内按下的 `n` / `→`，包括长按产生的**键盘自动重复**，会被全局
监听器抓到并**锁存**在 `events` 字典里。等下一条 episode 的 `record_loop`
一进来，第一次循环就命中 `exit_early` 直接 break —— 0 帧 —— 保存时抛异常。

**为什么以前碰不上**：没装 `pynput` 时走 `TerminalKeyListener`，只读当前
终端的 TTY，录制时焦点在别处，按键根本收不到，自然也锁存不了。
装上 pynput 换成全局监听后（见第 22 条），这个 bug 才暴露。

> **修好一个问题会让下一个问题显形。** 第 22 条让按键真的能用了，
> 第 23 条是"按键能用"之后才可能发生的事。这是推进，不是倒退。

**修复**：`lerobot_patch.py` 新增 `apply_exit_early_guard()`，两道防护：

| # | 位置 | 做法 |
|---|---|---|
| 1 | `record_loop` 入口 | **录制阶段**（`dataset is not None`）清掉残留的 `exit_early` |
| 2 | `LeRobotDataset.save_episode` | buffer 为空则跳过并告警，**不抛异常** |

第 1 道只在录制阶段清，**重置阶段（`dataset is None`）不清** —— 那里锁存的
按键是用户主动想跳过重置，应当保留。这个区分很重要，一刀切会让"跳过重置"失效。

第 2 道是兜底：万一还有别的路径产生空 episode，也不会让已录的几十条陪葬。

**操作上**：`save_episode()` 编码那几秒别按键、别按住不放。

**排查时的一个教训**：安装脚本里写了

```bash
$V/pip install ... 2>&1 | tail -6
```

`$V` 因为路径转换变成了空串，`/pip` 报 `No such file`（退出码 127），但
**管道的退出码取的是 `tail` 的 0**，于是 `set -e` 没拦住，脚本一路打印
"DONE" 假装成功。装完什么都没有。
**`cmd | tail` 会吞掉 cmd 的失败，检查安装结果不能只看最后那行 DONE。**

---

### 24. WSL 训练环境

Windows 只负责硬件（串口 + 相机），**训练全部在 WSL 跑**。两边互不干扰。

```
venv     ~/lerobot-train/.venv
lerobot  0.6.1
torch    2.11.0+cu130   CUDA 可用
GPU      RTX 3080 / 10 GB
```

**`.wslconfig` 必须改大**（原备份在 `~/.wslconfig.bak-*`）：

```ini
[wsl2]
memory=10GB      ; 原 4GB -- ACT 要同时解码两路视频喂 dataloader，4GB 会 OOM
processors=12    ; 原 4
swap=8GB         ; 原 2GB
```

宿主 16 GB / i7-11700KF 8C16T，给 WSL 10 GB 剩 6 GB 给 Windows。
**改完必须 `wsl --shutdown` 才生效。** 端口 8000 的 uvicorn 是随 WSL 自启的，
重启会自己回来。

**装的顺序**：先 `torch --index-url .../cu130`，再 `lerobot[dataset]`。
反过来的话 lerobot 会先拉一个 CPU 版 torch 下来，白下 2 GB。

**`torchcodec` 加载不了**（Windows 和 WSL 都是），缺对应版本的 FFmpeg 共享库。
自动退回 `pyav`，能用，只是解码慢一点。不值得为这个折腾。

**从 Windows 调 WSL 要用 PowerShell，不要用 Git Bash**：Git Bash 的 MSYS
路径转换会把 `/home/padiac/...` 改写成 `D:/Program Files/Git/home/padiac/...`，
表现是"变量没展开"之类的怪错。

---

### 25. 第一次训练：ACT / 30 条 / 100k 步

**数据**：`datasets/so101_pickplace`，30 episodes / 14332 frames / 8 分钟 / 251 MB，
单一 task 字符串 `Pick the yellow block and put it in the black bowl.`，
每条 315–722 帧（10.5–24.1 秒）。

**训练前必做：把数据集拷进 WSL 的 ext4。**

```bash
wsl -d Ubuntu -- cp -r /mnt/e/Repo/so101-build/datasets/so101_pickplace                        /home/padiac/lerobot-train/data/
```

训练要反复随机读小文件，走 `/mnt/e` 的 drvfs 慢一个数量级。

**先跑烟雾测试，不要直接开 100k 步。** 200 步、几十秒，能提前暴露所有依赖问题：

```bash
wsl -d Ubuntu -- /home/padiac/lerobot-train/.venv/bin/lerobot-train   --dataset.repo_id=local/so101_pickplace   --dataset.root=/home/padiac/lerobot-train/data/so101_pickplace   --policy.type=act --policy.device=cuda --policy.push_to_hub=false   --output_dir=/tmp/smoke --job_name=smoke   --steps=200 --save_freq=200 --log_freq=50   --batch_size=8 --num_workers=6 --wandb.enable=false
```

第一次跑就是靠它抓到 **缺 `lerobot[training]`（`accelerate`）** —— 这个包
`lerobot[dataset]` 不带。直接开 100k 步的话，几秒后崩，白等。

**实测数字（RTX 3080）**：

```
6 step/s   updt_s 0.167   data_s 0.003   mem_gb 3.73
loss 14.5 -> 3.6（200 步内）
```

`data_s 0.003` vs `updt_s 0.167` —— **瓶颈在 GPU 计算，不在数据加载**。
说明 `num_workers=6` 绰绰有余，也反证了拷进 ext4 这步做对了。
显存只用 3.73 GB，但 `batch_size=8` 是 ACT 预设调好的值，
改它要连学习率一起调，第一轮不动。

100000 步 = 约 56 个 epoch（14332 帧 / batch 8）= **约 4.6 小时**。

**`torchcodec` 在 WSL 里可以修好**，不用忍受 pyav：

```bash
sudo apt install -y ffmpeg
```

它报的 `libavdevice.so.60` 正是 Ubuntu 24.04 自带 FFmpeg 6.1 的版本号，
装上就能加载。日志里 `'video_backend': 'torchcodec'` 就是修好了。
（Windows 那边仍然是 pyav，缺 full-shared 版 FFmpeg 的 DLL，录制不受影响。）

#### 定时训练

`train_act.sh` 放在项目里（不在 WSL 里），Windows 任务计划程序调：

```powershell
$action  = New-ScheduledTaskAction -Execute "wsl.exe" `
           -Argument "-d Ubuntu -- bash /mnt/e/Repo/so101-build/train_act.sh"
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date "2026-08-27 03:00:00")
$set     = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable `
           -ExecutionTimeLimit (New-TimeSpan -Hours 12)
Register-ScheduledTask -TaskName "lerobot-act-train" `
           -Action $action -Trigger $trigger -Settings $set
```

比挂一个 `sleep` 进程可靠：`-WakeToRun` 能唤醒睡眠中的机器，
而 sleep 进程 WSL 一关就没了。取消：

```powershell
Unregister-ScheduledTask -TaskName "lerobot-act-train" -Confirm:$false
```

**挂上之后一定要验证那条命令行真能跑通。** 脚本里留了 `STEPS` 环境变量覆盖，
就是为了用少量步数走**完全相同的调用路径**：

```bash
wsl -d Ubuntu -- bash -c "STEPS=20 bash /mnt/e/Repo/so101-build/train_act.sh"
```

任务挂上了却调不起来，是这类定时任务最典型的失败方式，而且要等到第二天早上才发现。

#### 两个 shell 陷阱（都在这一轮踩到）

**`cmd | tail` 会吞掉 cmd 的失败**。管道的退出码取最后一个命令的，
`set -e` 拦不住，脚本会一路打印 "DONE" 假装成功。

**`set -euo pipefail` 下管道一失败就立刻退出**，后面取 `${PIPESTATUS[0]}`
的那行根本执行不到。要把真实退出码打进日志，得 `set +e` 包起来再 `set -e`。

---

### 26. 第一次部署：策略乱动不去夹 —— 排查过程与真正原因

**现象**：`robot.ps1 eval` 跑起来了，机械臂一会儿上一会儿下，不去夹东西。

这类问题最容易变成瞎调参。下面是逐个**用测量排除**的过程，每一条都留着，
因为排除掉的东西以后不用再怀疑第二次。

| 检查 | 工具 | 结果 | 结论 |
|---|---|---|---|
| 相机索引换位 | `check_cam_match.py` | margin 0.888，与 camera_map 一致 | 排除 |
| 单步预测精度 | `eval_offline.py` | MAE 1.0，占行程 0.8–1.5% | 模型准 |
| 开环误差累积 | `chunk_error.py` | step1→step100: 1.00→1.96 | 开环本身不烂 |
| `max_relative_target` 限幅 | `deploy_check.py` | 最大单步 delta 10.15，超限 0.00% | 从未生效 |
| 起始姿态偏离 | `deploy_check.py` / `home.py --dry-run` | 当前姿态已在分布内 | 排除 |
| 光照/外观域偏移 | `domain_gap.py` | 实时 vs 训练 0.860，训练内部基线 0.829 | 排除 |
| 实时观测下的规划 | `live_probe.py` | 净移 75、路径 82（path/net 1.1），夹爪张开 | **是连贯伸手** |

**真正原因：`n_action_steps=100`。**

ACT 一次预测 `chunk_size=100` 个动作。lerobot 的 checkpoint 默认
`n_action_steps=100`，即**这 100 步全部开环执行** —— 30 fps 下 3.3 秒不看画面。
30 秒的 eval 里只有 9 次修正机会。

第一段只要没走准，机械臂就到了**训练数据里没有的位置**。而 30 条数据全是
"顺利完成"的轨迹，里面**没有任何"歪了怎么回来"的样本**。下一段规划就是在
没见过的状态上外推，越走越偏。

> 这是行为克隆的经典失效模式：**复合误差 / 协变量偏移**。DAgger 就是为它发明的。
> 不是 bug，是这个范式的固有性质，跟数据干不干净无关。

**修法：缩短开环长度。**

```powershell
.
obot.ps1 eval -Duration 30 -ActionSteps 5    # 每 167ms 重规划
.
obot.ps1 eval -ActionSteps 1                 # 每帧重规划，余量小
```

这需要 GPU。CPU 上一次前向 **314 ms**，比 33 ms 的控制周期还长，闭环根本追不上。

#### Windows 装 CUDA torch

```powershell
.\.venv-win\Scripts\pip.exe install torch torchvision `
  --index-url https://download.pytorch.org/whl/cu130 --force-reinstall
```

实测 **314 ms -> 18.3 ms**（RTX 3080），17 倍。装完 torch 从 2.11 跳到 2.13，
**版本变了必须重验 lerobot 还能用**，别假设。

#### 两个反复踩到的调用陷阱

**① 必须走 pre/post processor。** lerobot 0.6 把归一化拆成了独立管线
（checkpoint 里的 `policy_preprocessor_*` / `policy_postprocessor_*`）。

```python
cfg = PreTrainedConfig.from_pretrained(CKPT)
pre, post = make_pre_post_processors(
    cfg, pretrained_path=CKPT,
    preprocessor_overrides={"device_processor": {"device": "cpu"}})
action = post(policy.select_action(pre(batch)))
```

漏掉这两步，喂进去的是未归一化的原始值，输出也没反归一化，
**MAE 会从 1.0 变成 55**，看起来像模型完全没训练。
第一次排查时就被这个误导过 —— 差点得出"训练失败"的错误结论。

**② 预处理管线里烘死了训练时的 `device=cuda`**，在 CPU 上加载会抛
`Requested device 'cuda' but CUDA is not available`。必须用
`preprocessor_overrides` 覆盖，`lerobot-rollout` 自己也是这么做的
（`rollout/context.py:471`）。

#### 起始姿态：这次不是主因，但机制要记住

`deploy_check.py` 量出 30 条数据的起始姿态极其集中
（`shoulder_lift` 的 min/max 只差 **0.35** 个单位）—— 策略从没见过别的起点。
这次恰好机械臂就停在那附近所以不是主因，但换个位置开跑就会是。

`home.py` 会在 eval 前把机械臂归位到该姿态（目标从数据集实时算，不写死），
`-NoHome` 可跳过。**任何开总线的脚本都要 `import lerobot_patch`**，
否则会撞上 `Lock` 写入丢包（第 1 条）。

---

### 27. ⭐ 真正的根因：22% 的数据在教它发呆

第 26 条把开环换成闭环之后，症状从"乱动"变成了"**基本不动**"。
这个变化本身是最大的线索。

`onset.py` 量出每条 episode 开头的静止时长：

```
最短  67 帧 (2.2 秒)
中位 100 帧 (3.3 秒)
最长 250 帧 (8.3 秒)
合计 3194 / 14332 帧 = 22.3% 的数据是机械臂坐着不动
```

按下开始录制之后，人会先调整姿势、看一眼、深吸一口气，**然后**才动手。
这几秒全被录进去了。

**问题不是"浪费了 22% 的数据"，是同一个画面配了两种互相矛盾的标签：**

| 帧 | 画面 | 标签 |
|---|---|---|
| 多数 | 静止的起始场景 | 继续等 |
| 少数 | **几乎一模一样**的静止场景 | 开始伸手 |

该等还是该动，**信息不在画面里，在操作者脑子里**。网络无法从图像区分，
而 ACT 最小化 L1，最优解就是输出两者的平均 —— "等"和"伸手"平均一下，
就是几乎不动。

这一条解释了两次 eval 的全部差异：

| `n_action_steps` | 现象 | 机制 |
|---|---|---|
| 100 | 乱动，不去夹 | 一次提交整段 100 步，块里后半段有真动作所以会动；但规划它的状态是歧义的 |
| 5 | **基本不动** | 每 167 ms 重做同一个"等"的决定，永远出不来 |

> **闭环没有让事情变糟，它把问题暴露了出来。** 之前开环"蒙头往前冲"
> 掩盖了策略在起始状态上根本没学到东西。
> 一个改动让症状变得更明显、更单一，通常是好事，不是倒退。

#### 修法：裁掉开头的静止段

`trim_dataset.py` —— 找到每条 episode 里动作真正开始的帧（任一关节
累计偏移超过 3 个单位），从**它前 5 帧**开始保留。

```bash
DATA=so101_pickplace_trim   # 训练时用这个
wsl -d Ubuntu -- /home/padiac/lerobot-train/.venv/bin/python     /home/padiac/lerobot-train/trim_dataset.py     --src .../so101_pickplace --dst .../so101_pickplace_trim --force
```

结果：**11288 帧保留，3044 帧删除（21.2%）**，静止段从 67/100/250 帧
降到 4/5/15 帧。

保留 5 帧引子是故意的 —— 让策略仍能看到一点起手过程，而不是从动作中途开始。

注意：**起始姿态的分布没变**（`shoulder_lift` spread 仍是 0.35），
这是对的。动作开始前 5 帧机械臂本来就还在原位。变的是**配对的标签**：
同一个姿态，后面跟着的现在一定是"动"。歧义消除在标签侧，不在状态侧。

#### 为什么重建而不是改元数据

v3 格式里 episode 通过 `from_timestamp` / `to_timestamp` 和
`dataset_from_index` / `dataset_to_index` 挂在**共享的视频文件**上，
还带着逐条的统计量（用于归一化）。手工维护这些不变量容易出错，
而且**错了是静默的** —— 数据集照样能加载，训练照样收敛，只是学错东西。
用官方 API 重建代价只是一次重编码，换来所有不变量由 lerobot 保证。

#### 采数据时怎么避免

> **模仿学习里，你演示的每一帧都是标签。** 在镜头前发呆的 3 秒，
> 网络会认认真真学成"这种情况下就该不动"。

- 手先放到位、想清楚要做什么，**再按开始**
- 或者接受它，事后用 `trim_dataset.py` 裁

这是遥操作采数据的通病，官方教程里不会提。

---

### 28. ⭐ 部署调参走到头之后：真正的瓶颈是数据的**结构**，不是数量

第一个策略从"原地抖"推进到"够到方块、差一点夹上"，全部靠部署层配置，**没有重训**。
但到此为止，配置这条路走完了。

#### 试过的全部配置（结论：都不是最终答案）

| 配置 | 结果 |
|---|---|
| `n_action_steps=100`（官方默认） | 会动，但乱走；裁剪后重测**无区别** |
| `n_action_steps=5` | **几乎不动**，只执行到轨迹慢起步段 |
| `n_action_steps=20` | 够到方块，但**举手又退回**（接缝把指令拽回落后的实测位置） |
| `+ temporal_ensemble_coeff=0.01` | 接缝消失，动作平滑，够到方块**差一点夹上**，然后停住 |
| CPU → CUDA | 前向 314ms → 18.3ms，闭环才跑得动 |
| `home.py` 归位 | 消掉一个变量，但当时姿态本就在分布内 |
| 裁掉开头静止段重训（第 27 条） | 起始状态规划路径 6.2 → 489.6，**这一步是真有效的** |

#### 真正的错误：我给的采集指导和官方相反

官方 [Tips for gathering data](https://huggingface.co/docs/lerobot/en/il_robots)：

> 至少录 **50 条**，**每个位置 10 条**。相机固定，**全程保持一致的抓取动作**。
> **不要太快引入太多变化，会拖累结果。**

而这一轮录的 30 条散在几十个不同位置上，**平均每个位置只有 1 条**。

结果完全对得上现象：**粗略伸手学会了（所有轨迹的共性），末端对不准（每个具体位置只见过一次）**。
这不是任何参数能修的。

> **数据的结构比数据的数量重要。** 30 条 × 30 个位置，远不如 50 条 × 5 个位置。
> 前者每个位置一个样本，学不出精度；后者每个位置十个样本，才能收敛到可重复的抓取。
> 变化要在**基线跑通之后**再加，不是一开始就铺满。

#### 官方还有一条自检，做之前先问自己

> **只看两路相机画面，你自己能完成这个任务吗？**

#### 下一轮的录制配方

```
5 个位置 × 每个位置 10 条 = 50 条
碗固定不动，相机别碰
每个位置内：同样的接近方式、同样的抓取姿势
手先放到位再按开始（避免开头发呆，见第 27 条）
```

**别在累的时候录。** 官方要求"全程保持一致的抓取动作"，而一致性正是疲劳时最先垮掉的东西。
模型能重训，参数能重调，**手抖录进去的就是抖的** —— 数据是这条链路上唯一不能事后补救的环节。

#### 这类问题是常见的，不是配置没配对

lerobot 仓库里有还开着的同症状 issue：
[#2213 Poor ACT Policy Performance After Training](https://github.com/huggingface/lerobot/issues/2213)、
[#1607 how to control a so-101 with trained ACT model?](https://github.com/huggingface/lerobot/issues/1607)。

社区反复强调的一句：**loss 收敛不预测真机成功率。** 本轮 loss 0.039 很漂亮，
和能不能抓到基本无关。别用 loss 判断该不该上机。

#### 留下的诊断工具（下次直接用，别再从头猜）

| 脚本 | 回答什么 |
|---|---|
| `robot.ps1 eval -Record` | 把 rollout 录成数据集，留下证据 |
| `analyze_rollout.py` | 观测在不在分布内 / 线上动作是否等于离线预测 / 抖动周期 |
| `tracking_error.py` | 指令 vs 实测：机械臂到底执行没执行 |
| `motion_profile.py` | 指令速度 vs 演示速度、反转周期 |
| `plan_agreement.py` | 各次重规划的意图方向一致吗 |
| `chunk_profile.py` | 动作块的前 N 步走了全程的百分之几 |
| `freeze_probe.py` | 卡住时它还想不想动 |
| `check_cam_match.py` | 相机索引有没有和训练时对调 |
| `wrist_compare.py` / `layout_ref.py` | 训练画面 vs 当前画面并排看 |

**注意 `analyze_rollout.py` 的 Q2 有局限**：它只比动作块的**第 1 步**，
而第 1 步永远约等于当前位置，所以"通过"是弱证据，不能据此认定部署管线无误。
要判断整段轨迹，用 `chunk_profile.py` 和 `plan_agreement.py`。

---

### 29. ⭐ 相机索引：不同 backend 的编号根本不是一回事

**现象**：`cams.py` 认定 `top=index 0`（俯视），lerobot 打开 index 0 却拿到了 Brio
（一个插来做语音输入的摄像头）。录制时 `observation.top` 显示的是人脸。

**根因**：两边用的 backend 不同。

| | DSHOW（`cams.py` 原来用的） | MSMF（lerobot 默认 `backend=ANY` 解析成的） |
|---|---|---|
| index 0 | 俯视 | **Brio** |
| index 1 | Brio | **俯视** |
| index 2 | 夹爪 | 夹爪 |

**索引不是设备的属性，是"某套枚举 API 眼中的排位"。** DirectShow 和 Media
Foundation 各枚举各的，顺序可以完全不同。一个脚本用 DSHOW 认出 index 0，
另一个用 MSMF 打开 index 0 —— **编号对上了，设备没对上。**

**修法：识别设备的工具，必须用消费它的代码所用的同一个 backend。**

```python
# cams.py
be = cv2.CAP_MSMF if sys.platform.startswith("win") else cv2.CAP_V4L2
```

```powershell
# robot.ps1，两处相机配置都要加
{ top: {type: opencv, index_or_path: N, ..., backend: 1400} }   # 1400 = Cv2Backends.MSMF
```

钉死之后识别相关度从 0.248（勉强）变成 **0.999**，Brio 被干净排除。

> **教训：任何按"编号"引用设备的地方，先问这个编号是谁给的。**
> 只要识别方和使用方不是同一套 API，编号就没有共同含义。
> 这类 bug 表现为"时好时坏、插拔一次就变"，很容易被误判成带宽、
> 驱动、缓存或者句柄没释放 —— 我在这上面依次试过这四种错误假设，
> 还去查了 USB 拓扑给"带宽不足"找支撑。**证据是假的时，后面的推理再严密也没用。**

**排查这类问题的正确起手式**：分别用每个 backend 拍一张带标签的联系表，
肉眼确认谁是谁，再谈其他。

```bash
python cams.py list        # 现在走 MSMF，和 lerobot 一致
python cams.py resolve     # 按画面内容认角色，不依赖编号
```

`cams.py resolve` 会扫所有索引、扔掉打不开和全黑的、用参考图匹配角色，
判不准就报错不写映射。`robot.ps1` 的 `record` 和 `eval` 每次都会自动跑它，
所以插拔任何摄像头都不需要手工改配置。

**注意参考图也要在同一个 backend 下重存** —— 旧的参考图是 DSHOW 时代存的，
存的可能就是错的设备。

---

### 30. ⭐ 部署参数 `n_action_steps` 造成的棘轮效应

**现象**：策略只会去第三个位置抓，方块放别处它照样去那儿，扑空后张着爪等到超时。
方块正好在那个点时，抓得又快又准。

**测量**：把 rollout 里真实发生过的观测按时间顺序重新喂给策略，看每次重规划的结果。

```
 t(s)   实际 pan    新计划起点    新计划终点
  0.0    -13.58       4.55        3.84
  0.5      5.23      10.90       11.34
  1.0     19.38      23.54       26.91
  2.0     43.21      47.23       47.06
  3.0     50.15      53.29       53.51
  5.0     54.81      54.68       54.75   <- 撞上训练分布远端，停住
```

**每一次重规划，新计划的起点都比当前位置再往前一点。** 动过去 → 重新规划 →
又往前一点 → 一路棘轮式推进到训练分布的最远端。

**但单个动作块自己不发散**：t=0 那个块是 `4.55 -> 最大 9.01 -> 终点 3.84`，
出去再收回来。**发散只发生在换块的接缝上。**

**修复：`n_action_steps` 15 -> 50（= chunk_size，整块执行）。**

```powershell
.
obot.ps1 eval -ActionSteps 50    # 已设为默认值
```

效果（用户实测）：从"只有位置 3、还得放得极准"变成"位置 2 和 3 都大致能成"，
位置 1 仍然够不到。**没有重新训练，只改了一个部署参数。**

> **`n_action_steps` 不是一个可以随便调小的性能旋钮。** 调小 = 更频繁地丢弃
> 计划并从当前位姿重新起算，而每个新计划都相对当前位姿有个小的正向偏移，
> 于是偏移会累积。ACT 论文里的动作分块本来就是要"提交一段时间"的。

`n_action_steps` 的上限是 `chunk_size`（烘进模型，训练时定），超了会报
`The chunk size is the upper bound for the number of action steps per model invocation.`

#### 参照物：别人实际用的配置

不要照抄博客散文里的数字，**去看别人上传到 HF 的模型里的 `train_config.json`**，
那是实际训练用的全部参数：

```
https://huggingface.co/<repo>/raw/main/train_config.json
```

两个已发布的 SO-101 ACT 模型（[ShubhamK32/act_so101_declutter](https://huggingface.co/ShubhamK32/act_so101_declutter)、
[Davidei/red_block_act](https://huggingface.co/Davidei/red_block_act)）用的都是 **lerobot 默认值**：

```
batch_size 8 · lr 1e-5 无调度器 · chunk_size 100 · n_action_steps 100
use_vae true · 480x640 不缩放 · ResNet18 ImageNet 预训练
```

而 v2 我照一篇博客把三处都改了：`chunk 50` / `n_action_steps 15` / `use_vae false`。
其中 `n_action_steps 15` 已被实测证明是有害的。

**教训：把"某篇博客写的数字"当标准之前，先找几个真实可用的配置文件看看主流是什么。**
一个人的配置可能是为他自己的约束（显存、时间）妥协出来的 —— ggando 缩到 224x224
是为了在 batch 64 下不爆显存，那是他的约束，不是任务的要求。

#### 内存：dataloader 缓冲必须随 batch 缩

试 `batch 64` 时把**整台机器**搞崩重启了。算一下就知道：

```
每样本 2 路 640x480x3 float ≈ 7.4 MB
batch 64 ≈ 470 MB
6 worker × 预取 4 × 470 MB ≈ 11 GB   > WSL 的 10 GB
```

`train_act.sh` 现在按 batch 自动缩（>=48 -> 2 worker / 预取 2）。
另外 batch 64 在 640x480 下 **GPU 也放不下**（10 GB 差 708 MB）。

---

### 31. ⭐ WSL 里的 `CUDA error: unknown error` —— 先查 Windows 事件日志，不要怀疑显卡

**现象**：100k 步训练跑到第 48349 步（2 小时 09 分）崩溃。

```
torch.AcceleratorError: CUDA error: unknown error
```

崩溃后 `nvidia-smi` 一切正常：47°C、空闲、驱动响应。

**真正的原因在 Windows 侧**：

```
21:04:06  Microsoft-Windows-Resource-Exhaustion-Detector  Event 2004
          vmmemWSL 消耗 15,720,484,864 字节 (14.6 GB)
          bambu-studio.exe 1.88 GB + 1.61 GB
          主机总内存 15.8 GB
21:04:39  训练崩溃
```

主机虚拟内存耗尽 → WSL 的 GPU 半虚拟化层（`dxgkrnl`）先失效 →
PyTorch 只能看到一个语焉不详的 CUDA 错误。**显卡从头到尾都是好的。**

**为什么 WSL 会涨到 14.6 GB**：`.wslconfig` 里两个键写错了段。

```ini
[wsl2]
autoMemoryReclaim=gradual   # WSL 2.1.5 在这个位置不认识它
sparseVhd=true              # 同上
```

这两个键是 **WSL 2.2.4** 才从 `[experimental]` 移到 `[wsl2]` 的。
本机是 **2.1.5.0**，写在 `[wsl2]` 下就是无效键（启动时有 `Unknown key` 警告，
一直被当成噪音忽略了）。**内存自动回收从未生效**：两小时里写了 12 GB
checkpoint、反复读数据集，page cache 只进不出。
`memory=10GB` + `swap=8GB` 允许虚拟机涨到 18 GB。

**修法**：

```ini
[wsl2]
memory=10GB
swap=2GB                    # 原来 8GB；上限 18 GB -> 12 GB

[experimental]              # 2.1.5 只认这个位置
autoMemoryReclaim=gradual
sparseVhd=true
```

改完 `wsl --shutdown` 重启，`Unknown key` 警告消失即为生效。

**教训**：

- WSL 训练遇到 `cudaErrorUnknown`，第一件事是查 Windows 事件日志 Event 2004，
  不是查显卡、驱动、CUDA 版本。
- **启动时的 `Unknown key` 警告不是噪音**，它意味着那条配置根本没生效。
- `.wslconfig` 的键在哪个段取决于 WSL 版本，跨版本抄配置会静默失效。
- 训练期间别开吃内存的桌面程序（这次是两个 Bambu Studio，合计 3.5 GB）。

**损失可控**：`save_freq=10000` 的 checkpoint 带 `training_state`，
只丢了 8349 步（约 22 分钟）。`train_resume.sh` 现在会自动从
`checkpoints/last` 续训并重试，且在"重启后步数没推进"时停下——
那说明不是瞬时故障，重试只会浪费一整晚。

```bash
bash train_resume.sh /home/padiac/lerobot-train/outputs/act_so101_v3_trim_20260901_1854
```


---

### 32. ⭐ 多任务策略的"段落感"：换指令必须先归位

`robot.ps1 eval -Live` 让指令跟着 `task.txt` 走，中途换句子不用重新加载模型
（`live_task.py`，原理是推理引擎每次推理都读一次 `self._task`）。

但光换掉那个属性，人在旁边**根本看不出机械臂在执行哪条指令**。三个原因叠在一起：

| 现象 | 原因 |
|---|---|
| 按下按钮后一两秒还在做上一件事 | 策略手里攒着一整块已经规划好的动作（`n_action_steps=50`，30fps 下 1.67 秒），队列空了才会去看指令 |
| 新指令一上来就乱伸 | 新指令是从**上一条指令把手臂丢下的地方**开始的——半途悬在别的方块上方，训练里从来没有哪一条 episode 从那种姿态开始 |
| 不知道到底发没发出去 | 面板只知道自己往文件里写了什么，没有任何东西回报机械臂正在执行什么 |

第二条最要命：它把"策略不听指令"和"策略起点在分布外"这两件完全不同的事
混在一起，只看机械臂是分不开的。

现在换成一个三状态机（`live_task.py` 里的 `_install_engine_patch`）：

```
idle  ──按下按钮──▶  homing  ──到位──▶  running  ──按下别的按钮──▶  homing ...
                    归位（每格 2.0 单位，容差 3.5，超时 5 秒）
```

按下按钮的那一刻就调 `engine.reset()`，把攒着的动作块直接丢掉，旧指令当场结束；
然后把手臂开回**训练数据的起始位姿**（和 `home.py` 同一个目标，
`LIVE_HOME_DATASET` 指到 checkpoint 自己训练用的那个数据集）；到位之后才开始新指令。
STOP 也一样：先归位，再停住。

每条指令因此有明确的开头和结尾，而且每一条都从分布内的起点开始。
实测归位约 29 tick（30fps 下 1.0 秒）。

面板多了一行状态，读的是策略端写出来的 `task.status`：

```
sent  red out
arm: returning to start pose ...      ← 黄色
arm: Pick the red cube out of the...  ← 绿色，这才是机械臂真正在执行的
```

两条线分开显示是有意的：**上面一行是你按了什么，下面一行是机械臂在干什么。**
两者不一致的那一两秒，正是以前误判策略的地方。

没设 `LIVE_HOME_DATASET`（或者策略是 nostate 版、帧里没有关节角）时会打印一行警告，
退回"立刻切换、不归位"的旧行为，而不是让整个 rollout 崩掉。

#### 32b. 清空 `task.txt` 不等于没有指令 —— `--task` 还在命令行上

第一版把 `task.txt` 写空就算数了，结果**什么都没按机械臂照样开始干活**，
干的还是那条旧的"把黄色方块放进碗里"。

原因是 `lerobot-rollout` 的 `--task` 是**整个 run 一个字符串**，没有"什么都不做"这个取值。
`robot.ps1` 即使在 `-Live` 模式下也照样把它的默认值传下去：

```powershell
[string]$Task = "Pick the yellow block and put it in the black bowl.",   # 第 55 行
...
--task="$Task"                                                          # 第 539 行
```

而 `live_task.py` 当时只在文件**非空**时才覆盖引擎的 `_task`：

```python
if _state["task"]:          # 空文件 -> None -> 不覆盖 -> 命令行的默认值原样留着
```

于是引擎带着"黄色放进碗里"出生，第一个 tick 就开始执行。
现在改成**文件永远赢，空文件也赢**，并打印一行说明忽略了哪个 `--task`。

教训：把一个"默认值"删掉,要顺着它的**所有**来源删。删了文件里的那份、
留下命令行里的那份，症状和一份都没删完全一样。

#### 32c. 做了不等于看得见

归位这套东西第一版是做了的，但终端上**一行都不打**，状态只写进网页读的那个文件。
从终端看，"归位跑了" 和 "归位根本没触发" 长得一模一样，等于没做。

现在每次状态转换都打一行，带数字：

```
[live-task] patched SyncInferenceEngine.get_action
[live-task] return-to-home pose from datasets\so101_mix1
[live-task] ignoring --task 'Pick the yellow block...'; idle until you send an instruction
[live-task] red out: broke off, homing (worst joint 55.0 units off)
[live-task] home reached in 0.9s (worst 3.0); now running red out
[live-task] STOP: broke off, homing (worst joint 50.0 units off)
[live-task] home reached in 0.8s (worst 2.0); now idle
```

第一行是关键：它证明补丁**确实挂在了 rollout 真正构造的那个引擎类上**。
之前的自测是往假模块里塞了个桩类，只证明了状态机的逻辑对，
没证明它在真实进程里跑得起来 —— 而后者才是有疑问的那个。

#### 32d. ⭐⭐ 归位走到一半就放弃 —— 速度按 tick 写，超时按秒写

症状：归位停在一个**不是起始位姿**的地方，然后策略从那儿开始，手腕相机被撞了好几次。

第一版这么写的：

```python
_HOME_STEP = 2.0        # 每 tick 最多走 2.0 单位
_HOME_TIMEOUT_S = 5.0   # 5 秒还没到就放弃
```

这两行只有在 30 Hz 下才是自洽的。实际的控制循环带着两个相机和一个策略，
跑出来是 **8.4 Hz**（rollout 自己会 WARN 这一行）：

| 循环频率 | 5 秒超时内允许走的距离 |
|---|---|
| 30 Hz | 300 单位 |
| 8.4 Hz | 84 单位 |

而实测所有录制帧到起始位姿的距离：

| 分位 | 距离 |
|---|---|
| p50 | 70 单位 |
| p90 | 131 单位 |
| p100 | 185 单位 |

**一半以上的姿态都走不完。** 归位半途超时 → 把一个半归位的、分布外的姿态交给策略
→ ACT 从这儿开 100 步开环 → 乱抡 → 撞相机。

不变量：**任何按 tick 计量的东西，量纲长度不由你控制。**
现在速度按秒写，每个 tick 用实测的 `dt` 换算成这一 tick 该走多远：

```python
_HOME_SPEED = 45.0        # 单位/秒，和 home.py 在 30fps 下的速率一致
dt = min(now - _ctl["tprev"], 0.25)
limit = min(_HOME_SPEED * dt, _HOME_MAX_STEP)
```

超时也不再是常数，而是**按这次实际要走的距离算**：`距离/速度 + 3 秒`。
固定值要么对长距离太紧，要么松到抓不住真卡住的关节。

还有一条：**超时之后不再启动策略。** 归位没走完就说明手臂在一个训练里没出现过的姿态上，
那正是乱抡的来源。现在进 `stuck` 状态原地不动，终端和面板都会说明是哪个关节差多少，
按 STOP 重试。

#### 32f. STOP 和 FREEZE 是两件事

"这一段结束了" 和 "它马上要撞上去了" 需要相反的行为，一个按钮做不了两件事：

| 按钮 | 行为 | 什么时候按 |
|---|---|---|
| **STOP** | 先归位回起始位姿，再停住 | 这条指令做完了，想干净地收尾 |
| **FREEZE** | 当拍停住，不归位，之后不发任何动作 | 手腕相机、桌沿、你的手，马上要挨上了 |

面板上 FREEZE 是红的、在 STOP 上面。`say.py` 里打 `freeze` / `急停` / `别动` 也可以。
FREEZE 之后按 STOP 就正常归位。

之所以要单独加：原来只有 STOP，而 STOP 会先归位 —— 那正是"眼看要撞上"时最不该做的事。

#### 32g. ⭐⭐ 归位的**路径**：不能一步到位，要先抬起再转手腕

撞坏东西的不是"归位"这个想法，是归位的**走法**。第一版所有关节同时朝目标插值，
在关节空间里画一条直线 —— 而这条直线会**穿过桌面，也会扫过手腕相机**。
已经因此撞到相机、把螺丝震松了。

现在分三段走，每一段只动它指名的关节，其余关节按实测位置**原地保持**：

| 段 | 动的关节 | 为什么 |
|---|---|---|
| 1 lift | shoulder_lift, elbow_flex | 先把手臂折起来，离开桌面 |
| 2 wrist | wrist_flex, wrist_roll | 抬起来之后再转手腕 |
| 3 home | 全部 | 最后才摆底座回起始位姿 |

每一段有自己按距离算的超时，不是整条路共用一个预算。
自测里从"手臂压低伸出、底座转出去 120、手腕翻过来"这个最坏姿态出发，实测顺序是：

```
shoulder_lift -> elbow_flex -> wrist_roll -> shoulder_pan
```

底座是最后动的。这条顺序现在是 `tools/policy/test_live_task.py` 里的一条断言。

**中间位姿可以自己录，不用猜角度**：抬多高、手腕停在哪个角度，取决于你桌上摆了什么，
数据集里读不出来。把手臂摆到你希望它经过的姿势，然后：

```powershell
.
obot.ps1 waypoint -Name so101_v3
```

存成 `home_waypoint.json`，前两段就走这个姿态。删掉文件就退回默认
（默认用起始位姿自己那几个关节的值，那本来就是折起来的休息姿态）。

**`home.py` 走的是同一套分段路径。** 这一点比 live 模式更要紧：
一次 run 一条指令之后，`home.py` 是**唯一**会带着手臂横穿桌面的东西，
而它每次 eval 开跑前都要从上一次 run 结束的姿态出发。
`home.py --save-waypoint` 就是上面那个 `robot.ps1 waypoint`。

#### 32g-2. ⭐ 真正该用的评测方式：一次 run 一条指令，30 秒到点自己结束

上面这一整套"运行中换指令"，起点是把 `-Duration` 设成一个很大的数字让它一直跑，
然后靠按钮切换。**这个前提本身就是错的。**

正确的做法简单得多：

```powershell
.
obot.ps1 eval -Name smolvla_mix1_100k -Task "Pick the red cube out of the bowl and put it on the table." -Duration 30
```

一次 run = 一次 trial。`-Duration` 默认就是 30。开跑前 `home.py` 归位，跑满 30 秒自己退出。
抓不到就算了，下一次 run 开始前会再归位。

好处不是省事，是**实验能读**：

- 每条指令拿到的时间预算完全一样，不同颜色、不同模型之间可比
- 整个 run 只有一条指令，不存在"这个动作是上一条还是这一条产生的"
- 中途没有姿态切换，也就没有"半归位的分布外起点"，没有乱抡

**不加 `-Live` 时，上面 32~32h 的所有补丁一个都不生效** ——
`live_task.apply()` 第三行就 return 了。所以这也是一个干净的 A/B：
怀疑是我改坏的，就去掉 `-Live` 跑一遍。

#### 32g-3. ⭐ 面板改成"一次按钮 = 一次 run"

`-Duration 30` 有个直接后果：**30 秒一到 rollout 就退出，再往 `task.txt` 里写什么都没人听了。**
所以只要还是"一个长跑的 rollout + 中途换指令"这个结构，30 秒的截止和连续测试就是矛盾的。
最早那版实现不了，不是参数没调对，是结构不对。

改法：面板不再写文件，而是**每按一次就起一个新的 run**。

```powershell
.\.venv-win\Scripts\python.exe panel.py --policy policies\smolvla_mix1_100k
```

| | |
|---|---|
| 按一下 | 起一个 `robot.ps1 eval -Policy ... -Task ... -Duration 30` |
| 跑的时候 | 所有按钮 disabled，显示 busy 和倒计时 |
| 30 秒到 | 进程自己退出，按钮重新可用 |
| 再按 | 起下一个 run，归位后重新开始 |

没有 STOP，没有中断。抓不到就是这一次 trial 的答案。
面板不传 `-Live`，所以 32~32h 那些补丁一个都不加载。

倒计时是从日志里 `control loop started` 那一行开始算的，不是从进程启动开始算：
加载 checkpoint、开两个相机、归位，上一次实测花了 7 秒，那不算 trial 的时间。
每次 run 的日志留在 `logs/eval_<时间戳>.log`。

`--stub` 用 sleep 代替真机，可以不接机械臂就把 busy 逻辑和倒计时点一遍。

#### 32h. 同一个按钮按第二次，什么都没发生

监视线程原来是比较**文本**变没变，一样就不推。于是"再按一次红色"是个空操作。
最要命的场合正是归位失败之后：面板写着"按一下重试"，而重试恰好是唯一没有效果的动作。

改成计写入次数（`_state["gen"]`），mtime 一变就 +1 并下推，
状态机比较的是 `(指令, 代数)`。按同一个按钮 = 重新开始这条指令，这才是按钮该有的语义。

#### 32e. 不用上机械臂就能验的自测

`tools/policy/test_live_task.py` —— 真实的 `SyncInferenceEngine` 类、
真实的 8.4 Hz 循环节奏、假的策略和假的舵机，一秒钟跑完四种情况：

```
.\.venv-win\Scripts\python.exe tools\policy	est_live_task.py
```

| 场景 | 断言 |
|---|---|
| 什么都没按 | 不发任何动作，`--task` 的默认值被忽略 |
| 130 单位外按 red | 2.9 秒归位到位，残差 < 3.5，然后才跑策略 |
| 185 单位外按 STOP | 4.2 秒归位到位，然后停住 |
| 某个关节卡住不动 | 放弃归位，**不跑策略**，原地不动 |
| 移动中按 FREEZE | 当拍停住，不归位，之后不再发任何动作 |
| FREEZE 之后按 STOP | 正常归位回起始位姿 |

前面那两个晚上真正浪费掉的，是"在机械臂上判断代码有没有跑"这件事本身。
能在一秒钟里判掉的，就不该拿一个晚上去判。

---

---

## 逐关节装配清单（含所有陷阱）

> 通用规则：**每个电机放进结构件之前先插好 3-pin 线**；装之前先确认这颗电机的 ID
> 对不对（贴标签）。

### Joint 1 — shoulder_pan (ID 1)
- 装**两个**盘。上盘用 1 颗 M3×6 固定，下盘不用螺丝
- 第一个电机放进 base
- 4 颗 M2×6 固定电机 —— ⚠️ **上面 2 颗、下面 2 颗**，不是一面 4 颗
- 滑上 first motor holder，两侧各 1 颗 M2×6
- 装 shoulder 件：**上面 4 颗 M3×6 + 下面 4 颗 M3×6**（共 8 颗）
- 装上 shoulder motor holder

### Joint 2 — shoulder_lift (ID 2)
- 装**两个**盘，上盘 1 颗 M3×6
- 第二个电机**从上方**滑入
- 4 颗 M2×6 固定
- 装 upper arm，**每侧 4 颗** M3×6

### Joint 3 — elbow_flex (ID 3)
- 装**两个**盘，上盘 1 颗 M3×6
- 插入电机 3，4 颗 M2×6 固定
- 装 forearm 到电机 3，**每侧 4 颗** M3×6

### Joint 4 — wrist_flex (ID 4)
- 装**两个**盘，上盘 1 颗 M3×6
- ⚠️ **顺序**：先滑上 motor holder 4，**再**滑入电机 4。反了装不进去
- 4 颗 M2×6 固定电机 4

### ⚠️⚠️ Joint 5 — wrist_roll (ID 5) —— 唯一的例外，看清楚
前四个关节都是"先装盘 → 再装电机"，**这里完全反过来**：

1. **先**把电机 5 插进 wrist holder，用 **2 颗**（不是 4 颗）M2×6 **前面**的螺丝固定
2. **然后**才装盘，而且**只装一个盘**（不是两个！），1 颗 M3×6 固定
3. 把 wrist 固定到电机 4，**两侧各 4 颗** M3×6

> **为什么**：腕部支架紧包着电机，盘先装上去外径超了，电机塞不进去。
> **踩过一次**（2026-08-16）：盘先装了，塞不进，只能拔盘。拔盘方法见踩坑记录 #7。

### Gripper（从臂）— gripper (ID 6)
- 把 gripper 装到电机 5，通过腕部那个盘固定，4 颗 M3×6
- 插入 gripper 电机，**每侧 2 颗** M2×6
- gripper 电机装**两个**盘，上盘 1 颗 M3×6
- 装 gripper claw，**两侧各 4 颗** M3×6

### Handle / Trigger（主臂）— 替代 Gripper
- leader holder 装到腕部，4 颗 M3×6
- handle 装到 leader holder，**1 颗 M2×6**
- 插入 gripper 电机，每侧 2 颗 M2×6，装 1 个盘 + 1 颗 M3×6 盘螺丝
- 装 follower trigger，4 颗 M3×6

---

## 装配注意事项（来自官方文档）

- **设 ID 必须在装配之前**。ID 写进舵机 EEPROM，一次搞定；但装好之后电机埋在结构件里，
  再想单独接线就得拆。
- 清支撑用小螺丝刀从底下撬，**电机槽位一定要清干净**，有残留装不进去。
- 每个舵机**放进结构件之前先插好 3-pin 线**。
- 舵机盘（motor horn）：上盘用 1 颗 M3×6 固定，下盘不用螺丝。
- 螺丝规格：M2×6（最小，固定电机本体，每个电机 4 颗）/ M3×6（固定结构件和舵机盘）。
- Waveshare 驱动板：两个跳线帽都要拨在 **B 通道（USB）**。
- 设 ID 时每按一次回车前**都要重新检查接线**，电源线很容易在操作中松脱。

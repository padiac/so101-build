#!/usr/bin/env python3
"""
LeRobot 健壮性补丁。

三个补丁，互相独立：
  A. enable_torque / disable_torque 加重试（舵机上电时序丢包）
  B. 丢弃 episode 开头锁存的 exit_early（按键把下一条打成 0 帧）
  C. 录制时按 b 标记「上一条已保存的 episode 是坏的」，收工后统一删

--- 补丁 A ---

背景：
    lerobot/motors/feetech/feetech.py

        def enable_torque(self, motors=None, num_retry: int = 0) -> None:
            for motor in self._get_motors_list(motors):
                self.write("Torque_Enable", motor, ENABLED, num_retry=num_retry)
                self.write("Lock", motor, 1, num_retry=num_retry)

    默认 num_retry=0。而 `torque_disabled()` 上下文退出时在 finally 里调用它。

    时序问题：逐颗写 Torque_Enable=1 会让舵机依次上电，12V 轨产生电流涌浪；
    紧跟着的 Lock=1 写入如果丢包，因为不重试就直接抛异常，整个连接失败。

    实测症状：
        Failed to write 'Lock' on id_=5 with '1' after 1 tries.
        [TxRxResult] There is no status packet!
    而串口本身是好的（probe_serial.py：300 次操作 0 失败，中位延迟 0.36ms）。

用法：在跑任何 lerobot 入口之前 import 并 apply：

    import lerobot_patch; lerobot_patch.apply()

或者直接用同目录的包装脚本 teleop_win.py / record_win.py。
"""

DEFAULT_RETRY = 5


def apply(num_retry: int = DEFAULT_RETRY, verbose: bool = True) -> None:
    from lerobot.motors.feetech.feetech import FeetechMotorsBus

    if getattr(FeetechMotorsBus, "_retry_patched", False):
        return

    _RETRY = num_retry
    orig_enable = FeetechMotorsBus.enable_torque
    orig_disable = FeetechMotorsBus.disable_torque

    # 参数名必须保持 num_retry —— lerobot 内部是用关键字调用的
    # (motors_bus.py: self.disable_torque(num_retry=5))，改名会导致
    #   TypeError: disable_torque() got an unexpected keyword argument 'num_retry'
    def enable_torque(self, motors=None, num_retry=0):
        return orig_enable(self, motors, num_retry=max(num_retry, _RETRY))

    def disable_torque(self, motors=None, num_retry=0):
        return orig_disable(self, motors, num_retry=max(num_retry, _RETRY))

    FeetechMotorsBus.enable_torque = enable_torque
    FeetechMotorsBus.disable_torque = disable_torque
    FeetechMotorsBus._retry_patched = True

    if verbose:
        print("[lerobot_patch] enable_torque / disable_torque num_retry -> {}".format(num_retry))

    apply_exit_early_guard(verbose=verbose)


def apply_exit_early_guard(verbose: bool = True) -> None:
    """补丁 B —— 丢弃在 episode 开始前锁存的 exit_early。

    背景：
        lerobot/scripts/lerobot_record.py 的 record_loop 把按键检查放在
        循环最顶端、加帧之前：

            while timestamp < control_time_s:
                if events["exit_early"]:
                    events["exit_early"] = False
                    break            # <- 一帧都没录就退出

        而 record() 主循环里 save_episode() 要编码 mp4，**耗时数秒**。
        这几秒内按下的 n / -> （包括长按产生的自动重复）会被全局监听器
        抓到并锁存在 events 里，等下一条 episode 的 record_loop 一进来
        就立刻 break，得到一个 0 帧的 episode，然后：

            ValueError: You must add one or several frames with `add_frame`
                        before calling `add_episode`.

        整个录制进程崩掉。

    为什么以前碰不上：
        没装 pynput 时走的是 TerminalKeyListener，只读当前终端的 TTY。
        录制时焦点在别的窗口，按键根本收不到，自然也就锁存不了。
        装上 pynput 换成全局监听后，这个 bug 才暴露出来。
        —— 修好一个问题会让下一个问题显形，这很正常。

    修法（两道）：
        1. record_loop 进入时，如果是**录制**阶段（dataset is not None）
           且 exit_early 已经是 True，说明这个按键是上一阶段的残留，清掉。
           重置阶段（dataset is None）不清 —— 那里锁存的按键是用户想跳过
           重置，应当保留。
        2. save_episode 兜底：buffer 为空就跳过并告警，不再抛异常。
           万一还有别的路径产生空 episode，也不会让已录的几十条陪葬。
    """
    import logging

    import lerobot.scripts.lerobot_record as rec

    if getattr(rec, "_exit_early_guard_patched", False):
        return

    orig_record_loop = rec.record_loop

    def record_loop(*args, **kwargs):
        events = kwargs.get("events")
        dataset = kwargs.get("dataset")
        # 只在录制阶段清；重置阶段的锁存是用户主动想跳过重置，要留着
        if events is not None and dataset is not None and events.get("exit_early"):
            events["exit_early"] = False
            logging.info("[lerobot_patch] discarded stale exit_early at episode start")
        return orig_record_loop(*args, **kwargs)

    rec.record_loop = record_loop

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    orig_save_episode = LeRobotDataset.save_episode

    def save_episode(self, episode_data=None, parallel_encoding: bool = True):
        if episode_data is None:
            buf = getattr(getattr(self, "writer", None), "episode_buffer", None)
            if buf is not None and not buf.get("size", 0):
                logging.warning(
                    "[lerobot_patch] empty episode buffer, skipping save "
                    "(no exception). Likely a stale exit_early at episode start."
                )
                return
        return orig_save_episode(self, episode_data, parallel_encoding)

    LeRobotDataset.save_episode = save_episode
    rec._exit_early_guard_patched = True

    if verbose:
        print("[lerobot_patch] stale exit_early guard enabled")


def apply_block_overlay(hsv_lo=(18, 90, 90), hsv_hi=(38, 255, 255), verbose=True):
    """Draw the yellow-block detection into Rerun, live, during a rollout.

    The block detector used for offline analysis is a plain HSV threshold, and a
    good deal has been concluded from it without ever watching it run. Overlaying
    it on the live feed makes it checkable while the arm is working: if the box
    jumps to a reflection or drops out while the block is plainly visible, that
    is visible immediately rather than buried in a statistic.

    ACT itself has no detector -- it maps pixels to actions end to end. This box
    is purely an instrument for us, and has no effect on what the policy sees.

    lerobot's strategies do `from ... import log_visualization_data`, so the name
    has to be replaced inside each module that imported it; patching the source
    module would have no effect on the already-bound names.
    """
    import logging

    import numpy as np

    try:
        import cv2
        import rerun as rr
    except ImportError as e:
        logging.warning("[lerobot_patch] block overlay unavailable: %s", e)
        return

    from lerobot.utils import visualization_utils as VU

    # Entity paths must match what rerun_visualization.py uses, which prefixes
    # every observation key: camera 'top' is logged at 'observation.top'.
    # Logging the box at 'top/block' put it outside the image view entirely,
    # which is why nothing appeared.
    from lerobot.utils.constants import OBS_PREFIX, OBS_STR

    orig = VU.log_visualization_data
    if getattr(VU, "_block_overlay_patched", False):
        return

    lo = np.array(hsv_lo, dtype=np.uint8)
    hi = np.array(hsv_hi, dtype=np.uint8)

    def detect(rgb):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        m = cv2.inRange(hsv, lo, hi)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not c:
            return None
        big = max(c, key=cv2.contourArea)
        return cv2.boundingRect(big), len(c)

    def _path(k):
        return k if str(k).startswith(OBS_PREFIX) else "%s.%s" % (OBS_STR, k)

    seen = set()

    def wrapper(display_mode, observation=None, action=None, compress_images=False):
        orig(display_mode, observation=observation, action=action,
             compress_images=compress_images)
        if display_mode != "rerun" or observation is None:
            return
        for key, val in observation.items():
            arr = np.asarray(val)
            if arr.ndim != 3 or arr.shape[2] != 3:
                continue
            try:
                r = detect(arr)
                if r is None:
                    rr.log("%s/block" % _path(key), rr.Clear(recursive=False))
                    continue
                (x, y, w, h), n = r
                if key not in seen:
                    seen.add(key)
                    logging.info("[lerobot_patch] drawing block box at %s/block", _path(key))
                rr.log("%s/block" % _path(key),
                       rr.Boxes2D(array=[[x, y, w, h]], array_format=rr.Box2DFormat.XYWH,
                                  labels=["block n=%d" % n]))
            except Exception:
                pass          # an overlay must never interrupt a rollout

    for mod in ("lerobot.rollout.strategies.core",
                "lerobot.rollout.strategies.episodic",
                "lerobot.scripts.lerobot_record",
                "lerobot.scripts.lerobot_teleoperate"):
        try:
            import importlib
            m = importlib.import_module(mod)
            if hasattr(m, "log_visualization_data"):
                m.log_visualization_data = wrapper
        except Exception:
            pass
    VU.log_visualization_data = wrapper
    VU._block_overlay_patched = True
    if verbose:
        print("[lerobot_patch] block bounding box will be drawn in Rerun")


# --- 补丁 C ---
#
# 背景：
#     lerobot 的 r / 左箭头只能重录「当前这一条」。等你发现上一条录砸了
#     （卡顿、手滑、物体没摆好），它已经 save_episode() 落盘了，而中途把它
#     从数据集里挖掉并不安全：streaming_encoding 下它的帧已经写进正在写的
#     mp4，从半截 mp4 里删一段不是原子操作。
#
#     所以这里不删，只记账：按 b 把「最后保存的那一条」的序号追加到数据集
#     目录下的 bad_episodes.txt，收工后用 robot.ps1 drop 一次性删掉——那条
#     路径已经有完整的校验和自动换名。

BAD_FILE = "bad_episodes.txt"
_last_saved = {"root": None, "index": None}


def _mark_last_episode_bad() -> None:
    """把最后保存的 episode 序号写进数据集目录下的 bad_episodes.txt。"""
    from pathlib import Path as _P

    root, idx = _last_saved["root"], _last_saved["index"]
    if root is None or idx is None:
        print("[bad] 还没有保存过 episode，无从标记")
        return
    f = _P(root) / BAD_FILE
    prev = [ln.strip() for ln in f.read_text().splitlines() if ln.strip()] if f.exists() else []
    if str(idx) in prev:
        print(f"[bad] episode {idx} 已经标记过了（当前共 {len(prev)} 条）")
        return
    prev.append(str(idx))
    f.write_text(chr(10).join(prev) + chr(10))
    print(f"[bad] 标记 episode {idx} 为坏；累计 {len(prev)} 条 -> {f}")


def apply_mark_bad(key: str = "b", verbose: bool = True) -> None:
    """按 `key` 标记上一条已保存的 episode 为坏。"""
    import lerobot.utils.keyboard_input as ki
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if getattr(ki, "_mark_bad_patched", False):
        return

    # 记住最后保存的是哪一条。放在 save_episode 之后取 num_episodes-1，
    # 因为那时计数已经加过了。
    _orig_save = LeRobotDataset.save_episode

    def save_episode(self, *a, **kw):
        out = _orig_save(self, *a, **kw)
        try:
            _last_saved["root"] = self.root
            _last_saved["index"] = self.num_episodes - 1
        except Exception:
            pass
        return out

    LeRobotDataset.save_episode = save_episode

    # create_key_listener 是在模块全局里被 init_keyboard_listener 查到的，
    # 所以替换模块属性就够，不需要动 init_keyboard_listener 本身。
    _orig_listener = ki.create_key_listener

    def create_key_listener(dispatch, *, controls_help: str = ""):
        def wrapped(name: str) -> None:
            if isinstance(name, str) and name.lower() == key:
                _mark_last_episode_bad()
                return
            dispatch(name)

        return _orig_listener(wrapped, controls_help=f"{controls_help}, {key}=mark last episode bad")

    ki.create_key_listener = create_key_listener
    ki._mark_bad_patched = True
    if verbose:
        print(f"[patch C] 录制时按 {key} 标记上一条 episode 为坏（写入 {BAD_FILE}）")

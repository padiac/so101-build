"""独立测试 lerobot 的按键监听 —— 不碰机器人，只验证输入层。

为什么单独测：录制跑起来牵扯串口、相机、Rerun 一大堆东西，
按键没反应时分不清是哪一层。这里只留输入这一个变量。

用法：python keytest.py
然后把窗口焦点切走（点浏览器/记事本都行），再按键。
全局监听正常的话，焦点在哪都能收到。
"""
import time

from lerobot.utils.keyboard_input import init_keyboard_listener, pynput_can_capture

print("pynput_can_capture =", pynput_can_capture(),
      "(True = 全局监听，焦点在哪都收得到)")

listener, events = init_keyboard_listener()
print("listener =", type(listener).__name__ if listener else None)
print()
print("按键测试中，30 秒。试试:")
print("  n 或 ->    下一条")
print("  r 或 <-    重录")
print("  q 或 Esc   停止")
print("请把焦点切到别的窗口再按，验证全局监听。")
print()

last = dict(events)
t0 = time.time()
while time.time() - t0 < 30:
    if events != last:
        print("  [%5.1fs] %s" % (time.time() - t0, events))
        # 消费掉，方便连按
        for k in events:
            events[k] = False
        last = dict(events)
    time.sleep(0.02)

print()
print("结束。全程一条都没打印 = 监听没生效。")
if listener:
    listener.stop()

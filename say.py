r"""Send instructions to a running policy, by typing or by voice.

The policy side is in live_task.py: `robot.ps1 eval -Live` starts a rollout whose
instruction follows a text file. This is the other half -- it writes that file.

    .\.venv-win\Scripts\python.exe say.py                     # type instructions
    .\.venv-win\Scripts\python.exe say.py --voice --lang zh   # speak them

The policy only ever saw six English sentences, so Chinese cannot be handed to it
directly -- it would be plainly out of distribution. The translation lives here
instead: Whisper hears Chinese, this maps it onto the sentence the policy knows,
and the policy stays untouched.
"""

import argparse
import os
import sys
from pathlib import Path

DEFAULT_FILE = Path(__file__).resolve().parent / "task.txt"

OUT_OF_BOWL = "Pick the {} cube out of the bowl and put it on the table."
INTO_BOWL = "Pick the yellow block and put it in the black bowl."
COLOURS = ["red", "blue", "yellow", "green", "purple", "pink"]

# Matching is by substring so a whole spoken phrase works, not just a bare word:
# "把红色的方块拿出来" contains 红, which is enough.
#
# Both character sets are listed because Whisper decides for itself which to emit
# and often picks traditional even for mainland speech -- "把紅色方塊拿出來" came
# back with 紅, missed a simplified-only table, and looked exactly like a model
# that had not understood.
ZH = {
    "red": ["红", "紅"],
    "blue": ["蓝", "藍"],
    "yellow": ["黄", "黃"],
    "green": ["绿", "綠"],
    "purple": ["紫"],
    "pink": ["粉"],
}
ZH_INTO_BOWL = [
    "放进碗", "放到碗", "放回碗", "装进碗", "放碗里",
    "放進碗", "放到碗", "放回碗", "裝進碗", "放碗裡",
]

SHORTCUTS = {c: OUT_OF_BOWL.format(c) for c in COLOURS}
SHORTCUTS["bowl"] = INTO_BOWL


def expand(text: str) -> str:
    """Map a colour word -- English or Chinese, bare or inside a sentence -- to the
    instruction the policy knows. Anything unrecognised is passed through."""
    raw = text.strip()
    key = raw.lower().rstrip(".")
    if key in SHORTCUTS:
        return SHORTCUTS[key]

    # "into the bowl" wins over a colour, because that sentence names a colour
    # too ("把黄色方块放进碗里").
    if any(k in raw for k in ZH_INTO_BOWL):
        return INTO_BOWL
    for colour, words in ZH.items():
        if any(w in raw for w in words):
            return OUT_OF_BOWL.format(colour)

    low = key
    if "bowl" in low and ("into" in low or "in the" in low):
        return INTO_BOWL
    for colour in COLOURS:
        if colour in low:
            return OUT_OF_BOWL.format(colour)
    return raw


def write(path: Path, task: str) -> None:
    path.write_text(task, encoding="utf-8")
    print(f"      -> {task}")


def type_loop(path: Path) -> None:
    print(f"writing to {path}")
    print(f"shortcuts: {', '.join(COLOURS)}, bowl, or the same in Chinese")
    print("ctrl-c to stop\n")
    while True:
        try:
            text = input("say> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        if text.lower() in ("q", "quit", "exit"):
            return
        if text.lower() in ("freeze", "急停", "别动", "別動"):
            # Stops the arm where it stands, without the return-to-home move that
            # STOP does. For when it is about to hit something.
            path.write_text("!freeze", encoding="utf-8")
            print("      -> FREEZE (stops moving now, stays where it is)")
            continue
        if text.lower() in ("stop", "idle", "停", "停下"):
            path.write_text("", encoding="utf-8")
            print("      -> idle (goes home, then holds still)")
            continue
        task = expand(text)
        if task == text.strip():
            print(f"      not one of the known instructions -- ignored")
            print(f"      known: {', '.join(COLOURS)}, bowl, stop")
            continue
        write(path, task)


def voice_loop(path: Path, model_size: str, device: str, lang: str | None = None,
               gate: float = 0.012) -> None:
    """Listen, and say out loud what was heard and what was made of it.

    The first version transcribed fixed three-second windows behind a silence
    gate and printed nothing when the gate held, so a mistuned threshold, a muted
    microphone and a model that heard nothing all looked identical from outside.
    Now the level meter is always on, an utterance is cut at its own trailing
    silence rather than on a timer, and every outcome prints: heard and matched,
    heard and not matched, or too quiet to reach the gate.
    """
    try:
        import numpy as np
        import sounddevice as sd
        from faster_whisper import WhisperModel
    except ImportError as e:
        print(f"missing dependency: {e.name}")
        print("install with:")
        print(r"  .\.venv-win\Scripts\python.exe -m pip install faster-whisper sounddevice")
        sys.exit(1)

    print(f"loading whisper ({model_size}, {device}) ...")
    model = WhisperModel(model_size, device=device,
                         compute_type="int8" if device == "cpu" else "float16")

    sr = 16000
    step = 0.1                   # seconds per read
    chunk = int(sr * step)
    release = 0.6                # seconds under the gate that end an utterance
    min_speech = 0.35
    max_len = 8.0

    dev = sd.query_devices(sd.default.device[0])["name"] if sd.default.device[0] is not None else "default"
    print(f"\nmic: {dev}")
    print(f"listening ({lang or 'auto'}), gate {gate:.4f}.  colours in English or Chinese, or 放进碗里.")
    print("the bar shows the live level -- if speaking never fills it, lower --gate.")
    print("ctrl-c to stop\n")

    buf: list = []
    quiet = 0.0
    peak = 0.0

    with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=chunk) as stream:
        while True:
            block, _ = stream.read(chunk)
            a = block[:, 0]
            level = float(np.abs(a).mean())

            bars = min(int(level / gate * 8), 20)
            tag = "REC " if buf else "    "
            sys.stdout.write("\r  " + tag + "[" + "#" * bars + " " * (20 - bars) + f"] {level:.4f}   ")
            sys.stdout.flush()

            if level >= gate:
                buf.append(a)
                quiet = 0.0
                peak = max(peak, level)
                if len(buf) * step >= max_len:
                    quiet = release
            elif buf:
                buf.append(a)
                quiet += step

            if not buf or quiet < release:
                continue

            audio = np.concatenate(buf)
            dur = len(audio) / sr
            hit = peak
            buf, quiet, peak = [], 0.0, 0.0
            sys.stdout.write("\r" + " " * 46 + "\r")

            if dur < min_speech + release:
                print(f"  (only {dur:.1f}s, peak {hit:.4f}) -- too short, ignored")
                continue

            segments, info = model.transcribe(audio, language=lang, beam_size=1, vad_filter=True)
            text = " ".join(s.text for s in segments).strip()
            detected = getattr(info, "language", "?")

            if not text:
                print(f"  {dur:.1f}s of sound (peak {hit:.4f}) but no words came out")
                continue

            task = expand(text)
            print(f'  [{detected} {dur:.1f}s] "{text}"')
            if task == text.strip():
                # Anything said near the microphone lands here -- a phone call, a
                # word to someone else, thinking aloud. Writing it through would
                # replace a working instruction with a string the policy has never
                # seen, and the arm would keep chasing it. Unrecognised speech is
                # dropped, and the standing instruction stays.
                print("      not an instruction -- ignored, still following the last one")
                continue
            write(path, task)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=Path(os.environ.get("LIVE_TASK_FILE", DEFAULT_FILE)))
    ap.add_argument("--voice", action="store_true")
    ap.add_argument("--model", default="small", help="small/base are multilingual; base.en is English-only")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--lang", default=None, help="zh, en, or omit to auto-detect")
    ap.add_argument("--gate", type=float, default=0.012, help="speech threshold; lower it if the bar never fills")
    ap.add_argument("task", nargs="*", help="send one instruction and exit")
    args = ap.parse_args()

    args.file.parent.mkdir(parents=True, exist_ok=True)
    if args.task:
        text = " ".join(args.task)
        task = expand(text)
        if task == text.strip():
            print(f'"{text}" is not one of the instructions this policy knows.')
            print(f"known: {', '.join(COLOURS)}, bowl -- or the same in Chinese")
            return 1
        write(args.file, task)
        return 0
    try:
        if args.voice:
            voice_loop(args.file, args.model, args.device, args.lang, args.gate)
        else:
            type_loop(args.file)
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

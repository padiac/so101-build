set -uo pipefail
B=/home/padiac/lerobot-train
"$B/.venv/bin/pip" install -q "lerobot[smolvla]" 2>&1 | tail -5
echo "--- versions ---"
"$B/.venv/bin/python" - <<'PY'
import importlib
for m in ("transformers","accelerate","torch","lerobot","num2words","safetensors"):
    try:
        mod = importlib.import_module(m)
        print(f"{m:<14} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"{m:<14} MISSING ({type(e).__name__})")
PY

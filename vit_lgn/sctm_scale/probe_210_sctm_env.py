import importlib
import sys

print("python", sys.executable, sys.version.split()[0])
for name in ["torch", "train_logic_vit_tiny", "goal6plus_vit_experiment", "probe_frozen_features"]:
    try:
        module = importlib.import_module(name)
        if name == "torch":
            print("torch", module.__version__, "cuda", module.cuda.is_available())
        else:
            print(name, "ok")
    except Exception as exc:
        print(name, "ERR", type(exc).__name__, exc)

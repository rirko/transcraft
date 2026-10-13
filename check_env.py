"""环境自检脚本：验证手搓 Transformer 所需的全部依赖。"""

import importlib
import importlib.metadata
import sys

PASS, FAIL = "✓", "✗"


def check(name: str, fn):
    try:
        info = fn()
        print(f"{PASS} {name}: {info}")
        return True
    except Exception as e:
        print(f"{FAIL} {name}: {type(e).__name__}: {e}")
        return False


def main() -> int:
    print(f"Python {sys.version.split()[0]} ({sys.executable})\n")

    ok = True

    import numpy
    ok &= check("numpy", lambda: numpy.__version__)

    import matplotlib
    ok &= check("matplotlib", lambda: matplotlib.__version__)

    import tqdm
    ok &= check("tqdm", lambda: tqdm.__version__)

    for mod in ("jupyter", "ipykernel", "tensorboard", "datasets", "tokenizers"):
        importlib.import_module(mod)
        ok &= check(mod, lambda m=mod: importlib.metadata.version(m))

    import torch
    ok &= check("torch", lambda: torch.__version__)
    ok &= check("CUDA available", lambda: torch.cuda.is_available())

    if torch.cuda.is_available():
        ok &= check("GPU", lambda: torch.cuda.get_device_name(0))
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        ok &= check("显存", lambda: f"{vram_gb:.1f} GB")

        # GPU 上跑一次小矩阵乘法
        a = torch.randn(512, 512, device="cuda")
        b = torch.randn(512, 512, device="cuda")
        c = a @ b
        torch.cuda.synchronize()
        ok &= check(
            "GPU 矩阵乘法",
            lambda: f"512x512 @ 512x512 -> {tuple(c.shape)}, mean={c.mean().item():.4f}",
        )

    print()
    print("全部通过，环境就绪！" if ok else "有检查项失败，请查看上方输出。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

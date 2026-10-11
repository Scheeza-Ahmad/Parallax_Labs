"""Required imports and a real CUDA operation. Failure always exits nonzero."""
import importlib
import json
import platform
from pathlib import Path


def main():
    result = {"python": platform.python_version(), "platform": platform.platform(), "imports": {}}
    passed = True
    for name in ["pandas", "sentence_transformers", "chromadb", "spacy", "torch"]:
        try:
            module = importlib.import_module(name)
            result["imports"][name] = {"ok": True, "version": module.__version__}
        except Exception as exc:
            passed = False
            result["imports"][name] = {"ok": False, "error": str(exc)}
    try:
        import torch
        result["cuda"] = {"available": torch.cuda.is_available(), "runtime": torch.version.cuda,
                          "device_count": torch.cuda.device_count()}
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: mandatory GPU requirement is BLOCKED")
        x = torch.tensor([2.0, 3.0], device="cuda")
        assert (x * x).sum().item() == 13.0
        torch.cuda.synchronize()
        result["cuda"].update(device=torch.cuda.get_device_name(0), operation_passed=True)
    except Exception as exc:
        passed = False
        result["gpu_error"] = str(exc)
    result["passed"] = passed
    Path("reports").mkdir(exist_ok=True)
    Path("reports/environment.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

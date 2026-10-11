"""Explicit runtime checks. Importing this module loads only the standard library."""
import importlib
from importlib import metadata
from pathlib import Path
import platform
import sys


def check_runtime(requirements, require_cuda=False):
    report = {"python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "checks": [], "packages": {}}

    def check(name, operation):
        try:
            result = operation()
            report["checks"].append({"name": name, "passed": True, "result": result})
        except Exception as error:
            report["checks"].append({"name": name, "passed": False,
                                     "error": f"{type(error).__name__}: {error}"})

    def python_version():
        if sys.version_info < (3, 12):
            raise RuntimeError("Python 3.12 or newer is required")
        return platform.python_version()

    check("python_version", python_version)
    for line in Path(requirements).read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, expected = line.split("==", 1)

        def pinned_version(name=name, expected=expected):
            actual = metadata.version(name)
            report["packages"][name] = actual
            # The CUDA wheel's local suffix is compatible with the public pin.
            if actual.split("+", 1)[0] != expected:
                raise RuntimeError(f"Expected {expected}, installed {actual}")
            return actual

        check(f"version:{name}", pinned_version)

    for name in ("torch", "transformers", "peft", "numpy", "pandas", "sklearn",
                 "sentencepiece", "tokenizers", "safetensors", "flask",
                 "label_studio_sdk", "pydantic", "semver", "colorama", "PIL",
                 "yaml", "requests", "tqdm"):
        check(f"import:{name}", lambda name=name: bool(importlib.import_module(name)))

    def backward(device, amp=False):
        import torch
        layer = torch.nn.Linear(64, 32).to(device)
        optimizer = torch.optim.AdamW(layer.parameters(), lr=0.001)
        before = layer.weight.detach().clone()
        inputs = torch.randn(16, 64, device=device)
        with torch.amp.autocast(device, enabled=amp):
            loss = layer(inputs).float().square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")
        if amp:
            scaler = torch.amp.GradScaler(device)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in layer.parameters()):
            raise RuntimeError("Missing or nonfinite gradients")
        if amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if torch.equal(before, layer.weight.detach()):
            raise RuntimeError("Optimizer did not update weights")
        if device == "cuda":
            torch.cuda.synchronize()
        return {"loss": loss.item(), "weights_updated": True, "amp": amp}

    check("cpu_backward_optimizer", lambda: backward("cpu"))

    def cuda_info():
        import torch
        available = torch.cuda.is_available()
        if require_cuda and not available:
            raise RuntimeError("CUDA is required but unavailable")
        if not available:
            return {"available": False}
        props = torch.cuda.get_device_properties(0)
        return {"available": True, "torch_cuda": torch.version.cuda,
                "name": props.name, "total_memory_bytes": props.total_memory,
                "capability": list(torch.cuda.get_device_capability(0))}

    check("cuda_device", cuda_info)
    cuda = next(c for c in report["checks"] if c["name"] == "cuda_device")
    if cuda.get("result", {}).get("available"):
        check("cuda_backward_optimizer", lambda: backward("cuda"))
        check("cuda_amp_backward_optimizer", lambda: backward("cuda", amp=True))
    report["passed"] = all(c["passed"] for c in report["checks"])
    return report

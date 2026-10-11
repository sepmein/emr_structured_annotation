"""Check the pinned training runtime; --help never imports ML libraries."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from training.environment import check_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New run directory; existing directories are refused")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    requirements = Path(__file__).with_name("requirements.txt")
    report = check_runtime(requirements, require_cuda=args.require_cuda)
    report.update(checked_at=datetime.now(timezone.utc).isoformat(),
                  requirements=str(requirements.resolve()),
                  requirements_sha256=hashlib.sha256(requirements.read_bytes()).hexdigest(),
                  require_cuda=args.require_cuda)
    root = Path(__file__).resolve().parents[1]
    report["code_sha256"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__).resolve(), Path(__file__).with_name("training") / "environment.py")
    }
    for key, command in (("git_commit", ["git", "rev-parse", "HEAD"]),
                         ("git_status", ["git", "status", "--porcelain"])):
        try:
            result = subprocess.run(command, cwd=root, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", check=True)
            report[key] = result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            report[key] = None
    target = args.output_dir / "runtime_check.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {target.resolve()}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

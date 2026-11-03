"""Entry point: python -m soup_lab demo|doctor|gpu-test|benchmark."""

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def doctor():
    import torch

    versions = {}
    for package in ("torch", "triton", "soup-cli", "transformers", "peft", "trl"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = {
        "python": sys.version.split()[0], "versions": versions,
        "cuda_available": torch.cuda.is_available(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "cpu_demo_available": True,
        "competition_gpu": "Kaggle Tesla T4; official harness remains authoritative",
    }
    print(json.dumps(result, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run a small CPU numerical experiment and save a report")
    demo.add_argument("--output", type=Path, default=ROOT / "reports" / "demo")
    demo.add_argument("--open", action="store_true", help="Open the local report in your browser")
    commands.add_parser("doctor", help="Print installed versions and GPU availability")
    commands.add_parser("gpu-test", help="Run actual CUDA kernel tests; fail if CUDA is absent")
    bench = commands.add_parser("benchmark", help="Run the unchanged official benchmark on a T4")
    bench.add_argument("--repeats", type=int, choices=[1, 2], default=2)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return doctor()
    if args.command == "demo":
        from .demo import run_demo

        try:
            report = run_demo(args.output)
        except FileExistsError as exc:
            parser.exit(2, f"{exc}\nChoose a new --output path to preserve previous evidence.\n")
        print(f"Report: {report}")
        print("CPU numerical demonstration only. No GPU speed or Soup quality claim.")
        if args.open:
            import webbrowser

            webbrowser.open(report.as_uri())
        return 0
    import torch

    if not torch.cuda.is_available():
        parser.exit(2, "CUDA is not available. Use the included Kaggle T4 notebook.\n")
    if args.command == "gpu-test":
        return subprocess.run([sys.executable, "-m", "pytest", "solution", "-q"], cwd=ROOT).returncode
    if not all("T4" in torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())):
        parser.exit(2, "The official benchmark requires Tesla T4.\n")
    for command in (
        ["task/prepare_data.py"],
        ["-m", "pytest", "solution", "-q"],
        ["harness/run_task.py", "--repeats", str(args.repeats)],
        ["harness/check_rules.py"],
        ["harness/score.py", "runs/result.json"],
    ):
        result = subprocess.run([sys.executable, *command], cwd=ROOT)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

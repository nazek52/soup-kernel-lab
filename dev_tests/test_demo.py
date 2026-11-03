import json
import subprocess
import sys

import pytest

from soup_lab.demo import run_demo


def test_demo_writes_measured_evidence_and_preserves_previous_run(tmp_path):
    out = tmp_path / "experiment"
    report = run_demo(out)
    data = json.loads((out / "result.json").read_text())
    assert len(data["cases"]) == 3
    assert all(case["passed"] for case in data["cases"])
    assert data["gpu_benchmark"] == "not run"
    assert report.exists()
    with pytest.raises(FileExistsError):
        run_demo(out)


def test_doctor_can_run_without_cuda():
    run = subprocess.run([sys.executable, "-m", "soup_lab", "doctor"], capture_output=True,
                         text=True, check=True)
    data = json.loads(run.stdout)
    assert data["cpu_demo_available"] is True
    assert "torch" in data["versions"]

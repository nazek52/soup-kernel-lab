"""Local completeness check, not a substitute for the official GPU harness or jury."""

import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]


def check(root=ROOT):
    errors = []

    def read(relative):
        try:
            return json.loads((root / relative).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            errors.append(f"Missing or invalid JSON: {relative}")
            return {}

    metadata = read("submission.json")
    if not isinstance(metadata, dict):
        return ["submission.json must be an object"]
    team = metadata.get("team")
    if not isinstance(team, list) or not 1 <= len(team) <= 4:
        errors.append("Fill team (1–4 members) in submission.json")
    elif any(not isinstance(person, dict) or not person.get("name") for person in team):
        errors.append("Each team member needs a name")
    for key in ("repository_url", "kaggle_saved_version_url", "video_url", "presentation_url"):
        value = metadata.get(key, "")
        parsed = urlparse(value if isinstance(value, str) else "")
        if parsed.scheme != "https" or not parsed.netloc:
            errors.append(f"Fill HTTPS URL: {key}")
        if key == "kaggle_saved_version_url" and value and (
            parsed.hostname not in ("kaggle.com", "www.kaggle.com") or not re.fullmatch(
                r"/code/[^/]+/[^/]+/?", parsed.path
            ) or not parse_qs(parsed.query).get("scriptVersionId", [""])[0].isdigit()
        ):
            errors.append("Kaggle URL must identify a saved code version (scriptVersionId)")
        if key == "repository_url" and value and parsed.hostname != "github.com":
            errors.append("repository_url must point to GitHub")
    commit = metadata.get("solution_commit", "")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        errors.append("Fill the full 40-character solution_commit SHA")

    for doc in ("README.md", "SUBMISSION.md"):
        text = (root / doc).read_text(encoding="utf-8")
        if "PENDING" in text:
            errors.append(f"Unfilled fields in {doc}")
        url = metadata.get("kaggle_saved_version_url")
        if url and url not in text:
            errors.append(f"Saved Kaggle URL missing from {doc}")

    result = read("runs/result.json")
    checks = read("runs/checks.json")
    if result and isinstance(checks, dict) and checks:
        digest = hashlib.sha256((root / "runs/result.json").read_bytes()).hexdigest()
        if checks.get("result_sha256") != digest:
            errors.append("Checks refer to a different result.json")
        rows = checks.get("checks")
        if not isinstance(rows, list) or len(rows) != 10 or not all(
            isinstance(row, dict) and row.get("ok") is True for row in rows
        ):
            errors.append("All ten official checks must pass")
        # Import the unchanged official checksum implementation, not a second algorithm.
        sys.path.insert(0, str(root / "harness"))
        from check_rules import solution_sha256

        if checks.get("solution_sha256") != solution_sha256(root / "solution"):
            errors.append("Solution changed since the official checks")
    return errors


if __name__ == "__main__":
    issues = check()
    for issue in issues:
        print(f"MISSING: {issue}")
    print("INCOMPLETE" if issues else "Local completeness checks passed; verify public access separately")
    sys.exit(1 if issues else 0)

#!/usr/bin/env python3
"""Run the smoke tests against GitHub main and email when results change.

Deployments run the code on GitHub main with this checkout's venv, so that is
what this tests: main is exported to a temporary folder and run with this
checkout's Python. Results are kept in the watchdog's state folder; an email
goes to the watchdog's recipients when a test starts failing or recovers.
"""

import argparse
import datetime as dt
import json
import os
import socket
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ops/watchdog"))
from prefect_watchdog import CONFIG_PATH, STATE_PATH, send_email  # noqa: E402

SMOKE_DATA = Path(os.environ.get("AMPLIFY_SMOKE_DATA", Path.home() / ".local/share/amplify-prefect/smoke"))
RESULTS_PATH = STATE_PATH.parent / "smoke.json"
TIMEOUT_HOURS = 4


def export_branch(dest, branch):
    subprocess.run(["git", "-C", str(REPO_ROOT), "fetch", "-q", "origin", branch], check=True)
    commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "FETCH_HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    archive = subprocess.Popen(["git", "-C", str(REPO_ROOT), "archive", "FETCH_HEAD"], stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(dest)], stdin=archive.stdout, check=True)
    if archive.wait() != 0:
        raise RuntimeError("git archive failed")
    return commit


def run_tests(code_dir, junit_path, extra_args):
    env = dict(os.environ, AMPLIFY_SMOKE_DATA=str(SMOKE_DATA))
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/smoke",
               f"--junitxml={junit_path}", *extra_args]
    return subprocess.run(command, cwd=code_dir, env=env, timeout=TIMEOUT_HOURS * 3600,
                          capture_output=True, text=True)


def read_junit(path):
    """{test name: None if it passed or was skipped, else a short failure message}."""
    results = {}
    for case in ET.parse(path).iter("testcase"):
        failure = case.find("failure")
        if failure is None:
            failure = case.find("error")
        message = None
        if failure is not None:
            message = (failure.get("message") or "").strip().splitlines()[0][:300] or "failed"
        results[case.get("name")] = message
    return results


def build_email(failures, previous, label, run_id):
    new = {t: m for t, m in failures.items() if t not in previous}
    recovered = sorted(t for t in previous if t not in failures)
    if not (new or recovered):
        return None
    parts = []
    if new:
        parts.append(f"{len(new)} smoke test{'s' if len(new) > 1 else ''} failing")
    if recovered:
        parts.append(f"{len(recovered)} recovered")
    sections = []
    if new:
        sections.append("NEWLY FAILING\n" + "\n".join(f"- {t}: {m}" for t, m in new.items()))
    still = {t: m for t, m in failures.items() if t in previous}
    if still:
        sections.append("STILL FAILING\n" + "\n".join(f"- {t}: {m}" for t, m in still.items()))
    if recovered:
        sections.append("RECOVERED\n" + "\n".join(f"- {t}" for t in recovered))
    sections.append(f"Full output: {run_id}")
    sections.append(f"-- run_smoke_tests on {socket.getfqdn()}, {label}")
    return ", ".join(parts), "\n\n".join(sections)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="print the email, send nothing, save no results")
    parser.add_argument("--branch", default="main", help="GitHub branch to test (default main); "
                        "results are only saved for main")
    # Anything else goes to pytest, e.g. -k onnx.
    args, pytest_args = parser.parse_known_args()

    config = json.loads(CONFIG_PATH.read_text())
    previous = json.loads(RESULTS_PATH.read_text()).get("failures", {}) if RESULTS_PATH.exists() else {}
    log_dir = STATE_PATH.parent / "smoke-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc)
    log_path = log_dir / f"{started:%Y%m%dT%H%M%S}.log"

    with tempfile.TemporaryDirectory(prefix="amplify-smoke-") as tmp:
        code_dir = Path(tmp) / "code"
        code_dir.mkdir()
        commit = export_branch(code_dir, args.branch)
        junit = Path(tmp) / "junit.xml"
        try:
            result = run_tests(code_dir, junit, pytest_args)
            log_path.write_text(result.stdout + result.stderr)
            results = read_junit(junit) if junit.exists() else {}
            if not results:
                tail = (result.stdout + result.stderr).strip().splitlines()[-1:] or ["no output"]
                results = {"(test run)": f"pytest exited {result.returncode}: {tail[0][:300]}"}
        except subprocess.TimeoutExpired:
            results = {"(test run)": f"timed out after {TIMEOUT_HOURS} h"}

    failures = {t: m for t, m in results.items() if m}
    passed = sorted(t for t, m in results.items() if not m)
    label = f"{args.branch} at {commit}"
    print(f"{started:%Y-%m-%d %H:%M} {label}: {len(passed)} passed or skipped, {len(failures)} failed")
    for test, message in failures.items():
        print(f"  FAILED {test}: {message}")

    email = build_email(failures, previous, label, log_path)
    if args.dry_run or args.branch != "main":
        print("\n".join(email) if email else "(no email: results unchanged)")
        return
    if email:
        send_email(config, *email)
    RESULTS_PATH.write_text(json.dumps({
        "finished": dt.datetime.now(dt.timezone.utc).isoformat(), "commit": commit,
        "passed": passed, "failures": failures, "log": str(log_path),
    }, indent=2))
    # Keep a month of logs.
    for old in sorted(log_dir.glob("*.log"))[:-30]:
        old.unlink()


if __name__ == "__main__":
    main()

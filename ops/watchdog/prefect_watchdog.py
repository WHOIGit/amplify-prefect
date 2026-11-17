"""Out-of-band health watchdog for a self-hosted Prefect stack.

Runs from cron every few minutes (see README.md next to this file). It checks
the systemd units, Prefect API, served deployments, flow runs, Postgres
container, registry pull access and disk space, then emails when a problem
appears, keeps going (daily reminder) or recovers, and for every flow run that
ends Failed or Crashed.

It deliberately does not depend on Prefect being healthy: if the API is down
that is itself reported, and the checks that need the API keep their last
known problems instead of reporting a false recovery.
"""

import argparse
import datetime as dt
import json
import os
import shutil
import smtplib
import socket
import subprocess
import traceback
from email.message import EmailMessage
from pathlib import Path

CONFIG_PATH = Path(os.environ.get("PREFECT_WATCHDOG_CONFIG", Path.home() / ".config/prefect-watchdog/config.json"))
STATE_PATH = Path.home() / ".local/state/prefect-watchdog/state.json"


def utcnow():
    return dt.datetime.now(dt.timezone.utc)


def fmt_age(delta):
    hours = delta.total_seconds() / 3600
    return f"{hours / 24:.1f} days" if hours >= 48 else f"{hours:.1f} h"


class Report:
    """Collects the results of one watchdog pass."""

    def __init__(self, previous_problems):
        self.problems = {}  # key -> message, for ongoing conditions
        self.events = []  # one-off messages, e.g. a failed flow run
        self._previous = previous_problems

    def problem(self, key, message):
        self.problems[key] = message

    def carry_over(self, prefix):
        """Keep last known problems for a check that could not run this time."""
        for key, info in self._previous.items():
            if key.startswith(prefix):
                self.problems.setdefault(key, info["message"])


def run(cmd, timeout=60):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def check_systemd(config, state, report):
    restarts = state.setdefault("restarts", {})
    for unit in config["systemd_units"]:
        out = run(["systemctl", "show", unit, "-p", "ActiveState,SubState,NRestarts,LoadState"])
        props = dict(line.split("=", 1) for line in out.stdout.splitlines() if "=" in line)
        if props.get("LoadState") != "loaded":
            report.problem(f"systemd:{unit}", f"{unit} is not loaded ({props.get('LoadState')})")
            continue
        if props.get("ActiveState") != "active":
            report.problem(
                f"systemd:{unit}",
                f"{unit} is {props.get('ActiveState')}/{props.get('SubState')}",
            )
        count = int(props.get("NRestarts") or 0)
        before = restarts.get(unit)
        if before is not None and count > before:
            report.events.append(
                f"{unit} was restarted by systemd {count - before} time(s) since the last check "
                f"(see: journalctl -u {unit} -n 100)"
            )
        restarts[unit] = count


def check_prefect(config, state, report):
    from prefect.client.orchestration import get_client
    from prefect.client.schemas.filters import (
        FlowRunFilter,
        FlowRunFilterEndTime,
        FlowRunFilterState,
        FlowRunFilterStateType,
        LogFilter,
        LogFilterFlowRunId,
    )
    from prefect.client.schemas.objects import DeploymentStatus, StateType
    from prefect.client.schemas.sorting import FlowRunSort, LogSort

    now = utcnow()
    ui = config["ui_url"].rstrip("/")

    with get_client(sync_client=True) as client:
        error = client.api_healthcheck()
        if error is not None:
            report.problem("prefect:api", f"Prefect API health check failed: {error}")
            report.carry_over("deployment:")
            report.carry_over("run:")
            return

        deployments = client.read_deployments()
        by_name = {d.name: d for d in deployments}
        by_id = {d.id: d for d in deployments}

        for name in config["deployments"]:
            deployment = by_name.get(name)
            if deployment is None:
                report.problem(f"deployment:{name}", f"Deployment {name} no longer exists")
            elif deployment.status != DeploymentStatus.READY:
                polled = deployment.last_polled
                age = f", last seen {fmt_age(now - polled)} ago" if polled else ""
                report.problem(
                    f"deployment:{name}",
                    f"Deployment {name} is NOT_READY: its serve process is not polling{age}. "
                    "New runs of it will not start.",
                )

        def describe(flow_run):
            deployment = by_id.get(flow_run.deployment_id)
            return (
                f"{flow_run.name} (deployment {deployment.name if deployment else 'none'})\n"
                f"    {ui}/runs/flow-run/{flow_run.id}"
            )

        # Flow runs that finished Failed/Crashed since the last pass. The first
        # pass only records a starting point so old failures are not re-sent.
        checked_until = state.get("runs_checked_until")
        seen = state.setdefault("seen_runs", {})
        if checked_until is not None:
            since = dt.datetime.fromisoformat(checked_until) - dt.timedelta(minutes=10)
            finished = client.read_flow_runs(
                flow_run_filter=FlowRunFilter(
                    state=FlowRunFilterState(
                        type=FlowRunFilterStateType(any_=[StateType.FAILED, StateType.CRASHED])
                    ),
                    end_time=FlowRunFilterEndTime(after_=since),
                ),
                sort=FlowRunSort.END_TIME_DESC,
                limit=200,
            )
            for flow_run in reversed(finished):
                key = f"{flow_run.id}:{flow_run.end_time.isoformat() if flow_run.end_time else ''}"
                if key in seen:
                    continue
                seen[key] = now.isoformat()
                message = (flow_run.state.message or "").strip() if flow_run.state else ""
                if len(message) > 1500:
                    message = message[:1500] + " ..."
                logs = client.read_logs(
                    log_filter=LogFilter(flow_run_id=LogFilterFlowRunId(any_=[flow_run.id])),
                    sort=LogSort.TIMESTAMP_DESC,
                    limit=15,
                )
                tail = "\n".join(
                    "      " + log.message.strip()[:500].replace("\n", "\n      ")
                    for log in reversed(logs)
                )
                report.events.append(
                    f"Flow run {flow_run.state_name}: {describe(flow_run)}\n    {message}"
                    + (f"\n    Last log lines:\n{tail}" if tail else "")
                )
        state["runs_checked_until"] = now.isoformat()
        cutoff = now - dt.timedelta(days=2)
        state["seen_runs"] = {
            k: v for k, v in seen.items() if dt.datetime.fromisoformat(v) > cutoff
        }

        # Runs that are stuck: orphaned, running too long, pending or late.
        active = client.read_flow_runs(
            flow_run_filter=FlowRunFilter(
                state=FlowRunFilterState(
                    type=FlowRunFilterStateType(
                        any_=[StateType.RUNNING, StateType.PENDING, StateType.SCHEDULED]
                    )
                )
            ),
            limit=200,
        )
        max_hours = config["running_max_hours"]
        for flow_run in active:
            deployment = by_id.get(flow_run.deployment_id)
            entered = flow_run.state.timestamp if flow_run.state else None
            in_state = now - entered if entered else dt.timedelta(0)
            key = f"run:{flow_run.id}"
            state_type = flow_run.state_type

            if state_type == StateType.RUNNING:
                limit = max_hours.get(deployment.name if deployment else "", max_hours["default"])
                if deployment and deployment.status != DeploymentStatus.READY:
                    report.problem(
                        key,
                        f"Running for {fmt_age(in_state)} but its serve process is down, "
                        f"so it is probably orphaned: {describe(flow_run)}",
                    )
                elif in_state > dt.timedelta(hours=limit):
                    report.problem(
                        key,
                        f"Running for {fmt_age(in_state)} (limit {limit} h): {describe(flow_run)}",
                    )
            elif state_type == StateType.PENDING:
                if in_state > dt.timedelta(minutes=config["pending_max_minutes"]):
                    report.problem(
                        key, f"Pending for {fmt_age(in_state)}: {describe(flow_run)}"
                    )
            elif state_type == StateType.SCHEDULED and flow_run.expected_start_time:
                late = now - flow_run.expected_start_time
                if late > dt.timedelta(minutes=config["late_max_minutes"]):
                    report.problem(
                        key, f"Scheduled but {fmt_age(late)} late: {describe(flow_run)}"
                    )


def check_postgres(config, state, report):
    name = config["postgres_container"]
    out = run(["docker", "inspect", "-f", "{{.State.Status}}", name])
    status = out.stdout.strip() or out.stderr.strip()
    if status != "running":
        report.problem(f"postgres:{name}", f"Postgres container '{name}' is not running: {status}")


def check_images(config, state, report):
    last = state.get("images_checked_at")
    if last and utcnow() - dt.datetime.fromisoformat(last) < dt.timedelta(
        minutes=config["image_check_minutes"]
    ):
        report.carry_over("image:")
        return
    for image in config["images"]:
        try:
            out = run(["docker", "manifest", "inspect", image], timeout=60)
        except subprocess.TimeoutExpired:
            report.problem(f"image:{image}", f"Timed out checking {image} in its registry")
            continue
        if out.returncode == 0:
            continue
        detail = (out.stderr or out.stdout).strip().splitlines()[-1:] or ["no output"]
        hint = ""
        if any(word in detail[0].lower() for word in ("denied", "unauthorized", "forbidden")):
            hint = (
                " The registry login in the service account's ~/.docker/config.json is probably expired "
                "or revoked: renew it, or 'docker logout <registry>' if the image is public."
            )
        report.problem(f"image:{image}", f"Cannot pull {image}: {detail[0]}.{hint}")
    state["images_checked_at"] = utcnow().isoformat()


def check_disks(config, state, report):
    for path in config["disk_paths"]:
        try:
            usage = shutil.disk_usage(path)
        except OSError as error:
            report.problem(f"disk:{path}", f"Cannot check disk {path}: {error}")
            continue
        percent = 100 * usage.used / usage.total
        if percent >= config["disk_max_percent"]:
            report.problem(
                f"disk:{path}",
                f"{path} is {percent:.0f}% full ({usage.free / 1e9:.0f} GB free)",
            )


CHECKS = [
    ("systemd", check_systemd, "systemd:"),
    ("prefect", check_prefect, None),
    ("postgres", check_postgres, "postgres:"),
    ("images", check_images, "image:"),
    ("disk", check_disks, "disk:"),
]


def send_email(config, subject, body):
    message = EmailMessage()
    message["From"] = config["email_from"]
    message["To"] = ", ".join(config["email_to"])
    message["Subject"] = f"{config.get('subject_prefix', '[prefect]')} {subject}"
    message.set_content(body)
    with smtplib.SMTP(config["smtp_host"], config["smtp_port"], timeout=30) as smtp:
        smtp.send_message(message)


def build_email(report, previous, now, remind_after):
    new = {k: m for k, m in report.problems.items() if k not in previous}
    recovered = {k: i["message"] for k, i in previous.items() if k not in report.problems}
    ongoing = {k: m for k, m in report.problems.items() if k in previous}
    reminder_due = any(
        now - dt.datetime.fromisoformat(previous[k]["notified"]) >= remind_after for k in ongoing
    )
    if not (new or recovered or report.events or reminder_due):
        return None

    parts = []
    if new:
        parts.append(f"{len(new)} new problem{'s' if len(new) > 1 else ''}")
    if report.events:
        parts.append(f"{len(report.events)} event{'s' if len(report.events) > 1 else ''}")
    if recovered:
        parts.append(f"{len(recovered)} recovered")
    if not parts:
        parts.append(f"{len(ongoing)} problem{'s' if len(ongoing) > 1 else ''} still open")
    subject = ", ".join(parts)

    sections = []
    if new:
        sections.append("NEW PROBLEMS\n" + "\n".join(f"- {m}" for m in new.values()))
    if report.events:
        sections.append("EVENTS\n" + "\n".join(f"- {m}" for m in report.events))
    if ongoing:
        lines = []
        for key, message in ongoing.items():
            since = dt.datetime.fromisoformat(previous[key]["since"])
            lines.append(f"- {message} (open for {fmt_age(now - since)})")
        sections.append("STILL OPEN\n" + "\n".join(lines))
    if recovered:
        sections.append("RECOVERED\n" + "\n".join(f"- {m}" for m in recovered.values()))
    sections.append(f"-- prefect_watchdog on {socket.getfqdn()} at {now:%Y-%m-%d %H:%M UTC}")
    return subject, "\n\n".join(sections)


def load_state():
    try:
        return json.loads(STATE_PATH.read_text())
    except FileNotFoundError:
        return {}


def save_state(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(STATE_PATH)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="print the email, send nothing, save no state")
    parser.add_argument("--test-email", action="store_true", help="send a test email and exit")
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text())
    if args.test_email:
        send_email(config, "watchdog test email", f"Test from prefect_watchdog on {socket.getfqdn()}.")
        return

    now = utcnow()
    state = load_state()
    previous = state.get("problems", {})
    report = Report(previous)

    for name, check, prefix in CHECKS:
        try:
            check(config, state, report)
        except Exception as error:
            report.problem(f"check:{name}", f"The {name} check itself failed: {error!r}")
            print(traceback.format_exc())
            if prefix:
                report.carry_over(prefix)
            else:
                report.carry_over("prefect:")
                report.carry_over("deployment:")
                report.carry_over("run:")

    # Events that could not be emailed last time are retried now.
    report.events = state.get("unsent_events", []) + report.events

    email = build_email(report, previous, now, dt.timedelta(hours=config["remind_hours"]))
    if args.dry_run:
        print(email[0] + "\n\n" + email[1] if email else "Nothing to report.")
        return

    sent = False
    if email:
        try:
            send_email(config, *email)
            sent = True
            print(f"{now:%Y-%m-%d %H:%M:%S} sent: {email[0]}")
        except Exception:
            print(traceback.format_exc())

    if email and not sent:
        # Leave problem state as it was so the same news is sent next time.
        state["unsent_events"] = report.events[-50:]
    else:
        state["unsent_events"] = []
        state["problems"] = {
            key: {
                "message": message,
                "since": previous.get(key, {}).get("since", now.isoformat()),
                "notified": now.isoformat() if email else previous[key]["notified"],
            }
            for key, message in report.problems.items()
        }
    save_state(state)


if __name__ == "__main__":
    main()

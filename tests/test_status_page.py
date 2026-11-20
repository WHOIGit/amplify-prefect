import datetime as dt
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "status_page", Path(__file__).resolve().parents[1] / "ops/watchdog/status_page.py"
)
status_page = importlib.util.module_from_spec(spec)
spec.loader.exec_module(status_page)

TODAY = dt.date(2026, 11, 20)


def test_component_of():
    assert status_page.component_of("prefect:api") == "Prefect server"
    assert status_page.component_of("systemd:prefect_server.service") == "Prefect server"
    assert status_page.component_of("systemd:prefect_postgres.service") == "Database"
    assert status_page.component_of("systemd:prefect_yolo_inference.service") == "Deployments"
    assert status_page.component_of("deployment:yolo-inference") == "Deployments"
    assert status_page.component_of("disk:/") == "Disk space"
    assert status_page.component_of("run:123") is None


def test_record_pass_and_uptime():
    history = {}
    for _ in range(3):
        status_page.record_pass(history, {}, TODAY)
    status_page.record_pass(
        history, {"deployment:a": "x", "deployment:b": "y", "systemd:prefect_a.service": "z"}, TODAY
    )
    day = history[TODAY.isoformat()]
    assert day["passes"] == 4
    # One down pass for the component, however many of its keys were open.
    assert day["down"] == {"Deployments": 1, "deployment:a": 1, "deployment:b": 1}
    assert status_page.uptime(history, "Deployments", 30, TODAY) == 0.75
    assert status_page.uptime(history, "Database", 30, TODAY) == 1
    assert status_page.uptime(history, "Database", 30, TODAY - dt.timedelta(days=1)) is None


def test_record_pass_prunes_old_days():
    history = {"2026-01-01": {"passes": 1, "down": {}}}
    status_page.record_pass(history, {}, TODAY)
    assert list(history) == [TODAY.isoformat()]


def sample_data(**overrides):
    now = dt.datetime(2026, 11, 20, 21, 0, tzinfo=dt.timezone.utc)
    data = {
        "generated": now.isoformat(),
        "ui_url": "https://prefect.example.org/prefect",
        "problems": [],
        "units": [{"unit": "prefect_server.service", "active": "active", "sub": "running", "restarts": 0}],
        "api_ok": True,
        "deployments": [{
            "name": "yolo-inference", "status": "NOT_READY", "last_polled": None, "uptime30": 0.5,
            "last_run": {"name": "run-1", "state": "Failed", "time": now.isoformat(), "url": "https://x/1"},
        }],
        "active_runs": [{"name": "<b>r</b>", "deployment": "d", "state": "Running",
                         "since": now.isoformat(), "url": "https://x/2"}],
        "recent_failed": [],
        "counts_24h": {"Completed": 2, "Failed": 1},
        "postgres": "running",
        "uptime": status_page.uptime_rows({}, TODAY),
        "incidents": [{"message": "m", "since": now.isoformat(), "until": now.isoformat()}],
        "gpus": [
            {"index": 0, "used": 3, "total": 46068, "util": 0, "use": "free",
             "flow_run_name": None, "flow_run_url": None},
            {"index": 1, "used": 9000, "total": 46068, "util": 80, "use": "flow",
             "flow_run_name": "run-9", "flow_run_url": "https://x/9"},
        ],
    }
    data.update(overrides)
    return data


def test_render_ok_and_escapes():
    page = status_page.render(sample_data())
    assert "All systems operational" in page
    assert "&lt;b&gt;r&lt;/b&gt;" in page and "<b>r</b>" not in page
    assert "Not Ready" in page
    assert "run-9" in page
    assert "<h2>Disk" not in page and "Container images" not in page
    assert page.count('<i class="none"') == 90 * len(status_page.COMPONENTS)


def test_render_problems():
    now = sample_data()["generated"]
    page = status_page.render(sample_data(problems=[{"key": "k", "message": "Disk <full>", "since": now}]))
    assert "1 open problem" in page
    assert "Disk &lt;full&gt;" in page


def test_write_page(tmp_path):
    status_page.write_page(tmp_path / "www", sample_data())
    assert (tmp_path / "www/index.html").read_text().startswith("<!doctype html>")
    data = json.loads((tmp_path / "www/status.json").read_text())
    assert len(data["uptime"][0]["days"]) == 90
    assert (tmp_path / "www/index.html").stat().st_mode & 0o777 == 0o644

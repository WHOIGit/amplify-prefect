"""Static status page for the Prefect stack, written by prefect_watchdog.py.

The watchdog gathers what it saw on each pass into a plain dict and calls
``write_page``, which renders it to ``index.html`` (plus ``status.json``) in the
configured directory for the web server to serve as static files. Uptime history
is a per-day count of watchdog passes and of the passes in which each component
had a problem, kept in the watchdog state by ``record_pass``.
"""

import datetime as dt
import html
import json
import os
from pathlib import Path

HISTORY_DAYS = 90

# Uptime bars, in display order. A pass counts against a component when any
# problem key starting with one of its prefixes is open.
COMPONENTS = [
    ("Prefect server", ("prefect:api", "systemd:prefect_server.service", "check:prefect")),
    ("Database", ("postgres:", "systemd:prefect_postgres.service")),
    ("Deployments", ("deployment:", "systemd:")),
    ("Image registries", ("image:",)),
    ("Disk space", ("disk:",)),
]


def component_of(key):
    for name, prefixes in COMPONENTS:
        if key.startswith(prefixes):
            return name
    return None


def record_pass(history, problem_keys, today):
    """Count one watchdog pass in ``history`` ({date: {"passes", "down"}})."""
    day = history.setdefault(today.isoformat(), {"passes": 0, "down": {}})
    day["passes"] += 1
    down = {component_of(key) for key in problem_keys} - {None}
    # Individual deployments too, for their own uptime column.
    down |= {key for key in problem_keys if key.startswith("deployment:")}
    for name in down:
        day["down"][name] = day["down"].get(name, 0) + 1
    cutoff = (today - dt.timedelta(days=HISTORY_DAYS)).isoformat()
    for date in [d for d in history if d <= cutoff]:
        del history[date]


def uptime(history, name, days, today):
    """Fraction of passes over the last ``days`` days with no problem, or None."""
    passes = down = 0
    for offset in range(days):
        day = history.get((today - dt.timedelta(days=offset)).isoformat())
        if day:
            passes += day["passes"]
            down += day["down"].get(name, 0)
    return None if passes == 0 else 1 - down / passes


def uptime_rows(history, today):
    rows = []
    for name, _ in COMPONENTS:
        days = []
        for offset in range(HISTORY_DAYS - 1, -1, -1):
            date = today - dt.timedelta(days=offset)
            days.append((date, uptime(history, name, 1, date)))
        rows.append({
            "name": name,
            "days": days,
            "pct30": uptime(history, name, 30, today),
            "pct90": uptime(history, name, HISTORY_DAYS, today),
        })
    return rows


# --- rendering ---------------------------------------------------------------

e = html.escape


def local(iso):
    if not iso:
        return ""
    when = dt.datetime.fromisoformat(iso).astimezone()
    return when.strftime("%b %-d, %H:%M %Z")


def age(iso, now):
    if not iso:
        return ""
    seconds = (now - dt.datetime.fromisoformat(iso)).total_seconds()
    if seconds < 90:
        return "just now"
    if seconds < 3600:
        return f"{seconds / 60:.0f} min"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


def pct(value):
    if value is None:
        return "no data"
    if value >= 0.99995:
        return "100%"
    return f"{100 * value:.2f}%"


def bar_class(value):
    if value is None:
        return "none"
    if value >= 0.9999:
        return "up"
    if value >= 0.99:
        return "minor"
    if value >= 0.95:
        return "partial"
    return "down"


def pill(text, kind):
    return f'<span class="pill {kind}">{e(text)}</span>'


def run_link(run):
    name = e(run["name"])
    return f'<a href="{e(run["url"])}">{name}</a>' if run.get("url") else name


def state_kind(state):
    return {
        "Completed": "ok", "Running": "info", "Pending": "info", "Scheduled": "muted",
        "Late": "warn", "Cancelled": "muted", "Cancelling": "muted", "Paused": "muted",
    }.get(state, "bad")


def render(data):
    now = dt.datetime.fromisoformat(data["generated"])
    problems = data["problems"]
    sections = []

    if problems:
        items = "".join(
            f'<li><span>{e(p["message"])}</span><small>since {e(local(p["since"]))}'
            f' ({e(age(p["since"], now))})</small></li>'
            for p in problems
        )
        title = f'{len(problems)} open problem{"s" if len(problems) != 1 else ""}'
        banner = f'<div class="banner bad"><strong>{title}</strong><ul class="problems">{items}</ul></div>'
    else:
        banner = '<div class="banner ok"><strong>All systems operational</strong></div>'

    # Uptime
    bars = []
    for row in data["uptime"]:
        cells = "".join(
            f'<i class="{bar_class(v)}" title="{date:%b %-d}: {pct(v)}"></i>' for date, v in row["days"]
        )
        bars.append(
            f'<div class="uptime-row"><div class="uptime-head"><span>{e(row["name"])}</span>'
            f'<span class="muted">{pct(row["pct30"])} (30 d) · {pct(row["pct90"])} (90 d)</span></div>'
            f'<div class="bars">{cells}</div></div>'
        )
    sections.append(
        '<section><h2>Uptime</h2>' + "".join(bars)
        + '<div class="bars-axis muted"><span>90 days ago</span><span>today</span></div></section>'
    )

    # Activity
    counts = data.get("counts_24h") or {}
    count_line = " · ".join(f"{n} {label.lower()}" for label, n in counts.items()) or "no runs"
    active = data.get("active_runs") or []
    rows = "".join(
        f'<tr><td>{run_link(r)}</td><td>{e(r["deployment"])}</td><td>{pill(r["state"], state_kind(r["state"]))}</td>'
        f'<td>{e(age(r["since"], now))}</td></tr>'
        for r in active
    ) or '<tr><td colspan="4" class="muted">Nothing running</td></tr>'
    failed = data.get("recent_failed") or []
    failed_rows = "".join(
        f'<tr><td>{run_link(r)}</td><td>{e(r["deployment"])}</td><td>{pill(r["state"], "bad")}</td>'
        f'<td>{e(local(r["end"]))}</td></tr>'
        for r in failed
    ) or '<tr><td colspan="4" class="muted">None</td></tr>'
    sections.append(
        f'<section><h2>Flow runs</h2><p class="muted">Last 24 hours: {e(count_line)}</p>'
        '<h3>Running now</h3><div class="scroll"><table><thead><tr><th>Run</th><th>Deployment</th><th>State</th><th>For</th></tr></thead>'
        f'<tbody>{rows}</tbody></table></div>'
        '<h3>Failed in the last 24 hours</h3><div class="scroll"><table><thead><tr><th>Run</th><th>Deployment</th><th>State</th><th>Ended</th></tr></thead>'
        f'<tbody>{failed_rows}</tbody></table></div></section>'
    )

    # Deployments
    rows = []
    for d in data.get("deployments") or []:
        status = pill("Ready", "ok") if d["status"] == "READY" else pill(d["status"].replace("_", " ").title(), "bad")
        last = d.get("last_run")
        last_cell = (
            f'{run_link(last)} {pill(last["state"], state_kind(last["state"]))}'
            f'<br><small class="muted">{e(local(last["time"]))}</small>'
            if last else '<span class="muted">never</span>'
        )
        rows.append(
            f'<tr><td>{e(d["name"])}</td><td>{status}</td><td>{pct(d.get("uptime30"))}</td><td>{last_cell}</td></tr>'
        )
    sections.append(
        '<section><h2>Deployments</h2><div class="scroll"><table><thead><tr><th>Deployment</th><th>Status</th>'
        '<th>Uptime (30 d)</th><th>Last run</th></tr></thead><tbody>'
        + ("".join(rows) or '<tr><td colspan="4" class="muted">Prefect API unavailable</td></tr>')
        + '</tbody></table></div></section>'
    )

    # GPUs
    if data.get("gpus") is not None:
        cards = []
        for g in data["gpus"]:
            if g["use"] == "flow":
                who = "Prefect: " + (run_link({"name": g["flow_run_name"] or "flow run", "url": g.get("flow_run_url")}))
                kind = "info"
                label = "In use by a flow"
            elif g["use"] == "other":
                who = "Outside Prefect"
                kind = "warn"
                label = "Busy"
            else:
                who = "Available to flows"
                kind = "ok"
                label = "Free"
            mem = 100 * g["used"] / g["total"] if g["total"] else 0
            cards.append(
                f'<div class="card"><div class="card-head"><strong>GPU {g["index"]}</strong>{pill(label, kind)}</div>'
                f'<div class="meter"><span style="width:{mem:.0f}%"></span></div>'
                f'<small class="muted">{g["used"] / 1024:.1f} / {g["total"] / 1024:.0f} GB · {g["util"]}% busy</small>'
                f'<div class="who">{who}</div></div>'
            )
        sections.append('<section><h2>GPUs</h2><div class="cards">' + "".join(cards) + '</div></section>')
    elif data.get("gpu_error"):
        sections.append(f'<section><h2>GPUs</h2><p class="muted">Could not read GPUs: {e(data["gpu_error"])}</p></section>')

    # Infrastructure
    unit_rows = "".join(
        f'<tr><td>{e(u["unit"].removesuffix(".service"))}</td><td>'
        + (pill("Running", "ok") if u["active"] == "active" else pill(f'{u["active"]}/{u["sub"]}', "bad"))
        + f'</td><td>{u["restarts"]}</td></tr>'
        for u in data.get("units") or []
    )
    disk_rows = "".join(
        f'<tr><td>{e(d["path"])}</td><td><div class="meter {"hot" if d["percent"] >= data["disk_max"] else ""}">'
        f'<span style="width:{min(d["percent"], 100):.0f}%"></span></div></td>'
        f'<td>{d["percent"]:.0f}%</td><td>{d["free_gb"]:,.0f} GB free</td></tr>'
        for d in data.get("disks") or []
    )
    image_rows = "".join(
        f'<tr><td>{e(i["image"])}</td><td>{pill("Pullable", "ok") if i["ok"] else pill("Cannot pull", "bad")}</td></tr>'
        for i in data.get("images") or []
    )
    sections.append(
        '<section><h2>Services</h2><div class="scroll"><table><thead><tr><th>systemd unit</th><th>State</th>'
        f'<th>Restarts</th></tr></thead><tbody>{unit_rows}</tbody></table></div></section>'
        '<section><h2>Disk</h2><div class="scroll"><table class="disk"><tbody>' + disk_rows + '</tbody></table></div></section>'
        '<section><h2>Container images</h2>'
        f'<p class="muted">Checked {e(local(data.get("images_checked")))}</p>'
        f'<div class="scroll"><table><tbody>{image_rows}</tbody></table></div></section>'
    )

    # Incidents
    incidents = data.get("incidents") or []
    items = "".join(
        f'<li><span>{e(i["message"])}</span><small>{e(local(i["since"]))} – {e(local(i["until"]))}'
        f' ({e(age(i["since"], dt.datetime.fromisoformat(i["until"])))})</small></li>'
        for i in incidents
    ) or '<li class="muted">None in the last 30 days</li>'
    sections.append(f'<section><h2>Resolved incidents</h2><ul class="problems">{items}</ul></section>')

    ui = e(data.get("ui_url") or "")
    return PAGE.format(
        css=CSS,
        generated=e(data["generated"]),
        generated_local=e(local(data["generated"])),
        ui=ui,
        banner=banner,
        body="\n".join(sections),
    )


def write_page(directory, data):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o755)
    serializable = dict(data)
    serializable["uptime"] = [
        {**row, "days": [[d.isoformat(), v] for d, v in row["days"]]} for row in data["uptime"]
    ]
    for name, content in (("index.html", render(data)), ("status.json", json.dumps(serializable, indent=1))):
        tmp = directory / f".{name}.tmp"
        tmp.write_text(content)
        os.chmod(tmp, 0o644)
        tmp.replace(directory / name)


CSS = """
:root{--bg:#f6f7f9;--panel:#fff;--text:#1d2330;--muted:#69707d;--line:#e3e6eb;
--ok:#1f9d55;--ok-bg:#e3f5ea;--bad:#c9302c;--bad-bg:#fbe7e6;--warn:#b7791f;--warn-bg:#fdf3dc;
--info:#2b6cb0;--info-bg:#e4eefa;--none:#d5d9df;--minor:#8fd19e;--partial:#f0b429}
@media (prefers-color-scheme:dark){:root{--bg:#12151b;--panel:#1b1f27;--text:#e6e8ec;--muted:#9aa1ad;
--line:#2c323d;--ok:#48c78e;--ok-bg:#16332a;--bad:#f2706b;--bad-bg:#3a1d1d;--warn:#f0b429;--warn-bg:#382d14;
--info:#6fa8ff;--info-bg:#1a2a44;--none:#353b46;--minor:#3f8f5c;--partial:#b7861a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 48px}
header{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;gap:8px;margin-bottom:16px}
h1{font-size:22px;margin:0}h2{font-size:17px;margin:0 0 12px}h3{font-size:14px;margin:16px 0 6px;color:var(--muted)}
a{color:var(--info)}
.muted{color:var(--muted)}
section{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;margin-top:16px}
.banner{border-radius:10px;padding:14px 16px;border:1px solid}
.banner.ok{background:var(--ok-bg);border-color:var(--ok);color:var(--ok)}
.banner.bad{background:var(--bad-bg);border-color:var(--bad);color:var(--text)}
.banner.bad strong{color:var(--bad)}
.banner.stale{background:var(--warn-bg);border-color:var(--warn);margin-bottom:12px}
.problems{margin:8px 0 0;padding:0;list-style:none}
.problems li{padding:8px 0;border-top:1px solid var(--line);display:flex;flex-direction:column;gap:2px;overflow-wrap:anywhere;white-space:pre-line}
.problems li:first-child{border-top:0}
.problems small{color:var(--muted)}
.uptime-row{margin-bottom:14px}
.uptime-head{display:flex;justify-content:space-between;flex-wrap:wrap;gap:4px;margin-bottom:4px}
.bars{display:flex;gap:1px;height:26px}
.bars i{flex:1;min-width:0;border-radius:2px;background:var(--none)}
.bars i.up{background:var(--ok)}.bars i.minor{background:var(--minor)}
.bars i.partial{background:var(--partial)}.bars i.down{background:var(--bad)}
.bars-axis{display:flex;justify-content:space-between;font-size:12px}
.scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse}
th,td{text-align:left;padding:7px 8px;border-top:1px solid var(--line);vertical-align:top}
th{font-size:12px;color:var(--muted);font-weight:600;border-top:0}
td{overflow-wrap:anywhere}
.pill{display:inline-block;font-size:12px;font-weight:600;padding:1px 8px;border-radius:999px;white-space:nowrap}
.pill.ok{background:var(--ok-bg);color:var(--ok)}.pill.bad{background:var(--bad-bg);color:var(--bad)}
.pill.warn{background:var(--warn-bg);color:var(--warn)}.pill.info{background:var(--info-bg);color:var(--info)}
.pill.muted{background:var(--line);color:var(--muted)}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px}
.card{border:1px solid var(--line);border-radius:8px;padding:12px}
.card-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.who{margin-top:6px;font-size:14px;overflow-wrap:anywhere}
.meter{height:8px;background:var(--line);border-radius:4px;overflow:hidden;min-width:80px}
.meter span{display:block;height:100%;background:var(--info)}
.meter.hot span{background:var(--bad)}
table.disk td:nth-child(2){width:45%}
"""

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="60">
<title>Prefect Status</title>
<style>{css}</style>
</head>
<body>
<main>
<header>
<h1>Prefect status</h1>
<span class="muted">Updated <time id="generated" datetime="{generated}">{generated_local}</time> · <a href="{ui}">Open Prefect</a></span>
</header>
<div id="stale" class="banner stale" hidden><strong>This page is out of date.</strong> The watchdog that updates it has not run since <span id="stale-age"></span>, so the status below may be wrong.</div>
{banner}
{body}
</main>
<script>
(function () {{
  var t = new Date(document.getElementById("generated").getAttribute("datetime"));
  var minutes = (Date.now() - t.getTime()) / 60000;
  if (minutes > 15) {{
    document.getElementById("stale").hidden = false;
    document.getElementById("stale-age").textContent = t.toLocaleString();
  }}
}})();
</script>
</body>
</html>
"""

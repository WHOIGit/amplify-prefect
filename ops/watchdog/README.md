# Prefect watchdog

`prefect_watchdog.py` runs every 5 minutes from the Prefect service account's
crontab on the Prefect host and emails `email_to` in its config (sent through the
local mail relay).

The config is not in the repo: copy `config.example.json` to
`~/.config/prefect-watchdog/config.json` (or point `PREFECT_WATCHDOG_CONFIG` at it)
and fill in the addresses and URL.

What it checks:

- **systemd units** in `systemd_units` are active, and notes when systemd restarted one
- **Prefect API** responds to its health check
- **deployments** in `deployments` are `READY`, meaning their `.serve()` process is polling
- **flow runs**: every run that ends Failed or Crashed (with its last log lines); runs
  that are orphaned (Running but their serve process is down), Running longer than
  `running_max_hours`, Pending longer than `pending_max_minutes`, or Scheduled and late
- **Postgres** container is running
- **images**: each image in `images` can be pulled from its registry with the stored credentials (hourly)
- **disk** usage of `disk_paths` is below `disk_max_percent`

It also writes a status page (`index.html` and `status.json`) to `status_page_dir`
on every pass: open problems, 90-day uptime, running and failed flow runs,
deployments, GPU use and services. `apache-status-page.conf` serves it
at `/prefect-status/`; installing it needs root. The page warns when it is more than
15 minutes old, which means the watchdog itself has stopped.

It emails when a problem appears, once a day while a problem stays open, when a
problem recovers, and for each failed run. State is kept in
`~/.local/state/prefect-watchdog/state.json` and output in `cron.log` next to it.

The script runs from this working copy, not from GitHub, so edits here take
effect on the next cron run.

## Install

```
crontab -e
*/5 * * * * flock -n /tmp/prefect-watchdog.lock /path/to/amplify-prefect/.venv/bin/python /path/to/amplify-prefect/ops/watchdog/prefect_watchdog.py >> $HOME/.local/state/prefect-watchdog/cron.log 2>&1
```

## Useful commands

```
.venv/bin/python ops/watchdog/prefect_watchdog.py --dry-run     # print what it would send
.venv/bin/python ops/watchdog/prefect_watchdog.py --test-email  # check mail delivery
.venv/bin/python ops/watchdog/prefect_watchdog.py --dry-run --page-dir /tmp/page  # preview the page
```

When you add or retire a served flow, update `systemd_units`, `deployments` and
`images` in the config, and `FLOW_UNITS` in `ops/systemd/install.sh`.

Not covered: if the host itself is down or offline, nothing here can send mail. An
external check (for example an uptime monitor on the server's `/api/health`)
would cover that.

# Smoke tests

`tests/smoke/` runs each flow for real: it pulls the real images, starts real
containers on real GPUs, and checks the outputs, using a few small real inputs
(a handful of IFCB bins, a few images, the production models). This catches
what the faked tests in `tests/flows/` can't: an image that is broken or whose
command line changed, a registry that is down, permission problems, CUDA
problems.

They take about 10 minutes and need the Prefect host, so CI doesn't run them.

## Sample data

The tests read `AMPLIFY_SMOKE_DATA` (by default
`~/.local/share/amplify-prefect/smoke` for the nightly run). Its layout is
described in `tests/smoke/conftest.py`. It is not in the repo; the folder's own
`SOURCES.sh` rebuilds it from the originals on the host. `smoke.json` there
names the inputs that are used in place rather than copied (Stingray video,
VAST tables, the dojo training lists, the scratch bucket the zip test uploads
to). A test whose section is missing from `smoke.json` is skipped.

Flow runs go to a temporary Prefect server, so they don't appear in the UI, but
GPU locks are the real ones: a GPU test waits for a free GPU and is skipped if
none frees up. The S3/VAST credentials are copied from the real server's
`vasts3-creds` block.

## Nightly run

`run_smoke_tests.py` exports GitHub main to a temporary folder (the code the
deployments run), runs the smoke tests with this checkout's venv (the venv the
deployments use), and emails the watchdog's recipients when a test starts
failing or recovers. Results go to `~/.local/state/prefect-watchdog/smoke.json`
and the full output to `smoke-logs/` next to it.

```
crontab -e
30 2 * * * flock -n /tmp/amplify-smoke.lock /path/to/amplify-prefect/.venv/bin/python /path/to/amplify-prefect/ops/smoke/run_smoke_tests.py >> $HOME/.local/state/prefect-watchdog/smoke-cron.log 2>&1
```

The venv needs pytest (`VIRTUAL_ENV=$PWD/.venv uv pip install pytest`).

## Running by hand

```
AMPLIFY_SMOKE_DATA=~/.local/share/amplify-prefect/smoke .venv/bin/python -m pytest tests/smoke -k onnx
.venv/bin/python ops/smoke/run_smoke_tests.py --dry-run                 # test main, print the email, send nothing
.venv/bin/python ops/smoke/run_smoke_tests.py --branch my-branch -k onnx  # test a branch before merging
```

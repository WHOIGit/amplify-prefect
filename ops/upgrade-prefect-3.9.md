# Upgrade the Prefect host from 3.6.27 to 3.9.3

Upgrades the server and every flow-serving process (all run from
`/opt/amplify-prefect/.venv`), runs the 3.9 database migrations, and moves
Postgres from 17.5 to 17.11. Expect about 10 minutes of downtime.

Rehearsed on 2026-11-17:

- the 34 tests pass on the new environment
- every flow module imports, except `seggpt_inference`, which has a bad import on 3.6.27 too
- `.serve()` runs a flow to Completed, and `onnx-inference` served from GitHub comes up READY
- the migrations ran cleanly on a copy of the production database (log/event rows
  left out), and a 3.9.3 server on that copy showed all deployments, runs and blocks

Steps marked **(root)** need sudo. Everything else runs as the Prefect service account from
`/opt/amplify-prefect`.

## 0. Before you start

- **Running flow runs are killed** when their serve process stops. Check the UI for
  Running runs and wait for them, or plan to re-run them afterwards.
- Check out the `security/prefect-3.9-deps` branch, so `src/requirements.txt` and
  `docker-compose.yml` are the new versions.

## 1. Back up the database

```
mkdir -p ~/backups
docker exec postgres pg_dump -U postgres -d prefect -Fc > ~/backups/prefect-3.6.27-$(date +%F).dump
ls -lh ~/backups/
```

## 2. Stop Prefect (root)

```
sudo systemctl stop prefect_onnx_inference prefect_slim_features prefect_yolo_inference \
  prefect_stingray_frame_timestamps prefect_stingray_image_abundance
sudo systemctl stop prefect_server
```

## 3. Postgres 17.5 to 17.11

This is a minor version upgrade with the same data directory, so no dump or restore is needed.

```
docker compose up -d
docker exec postgres psql -U postgres -tAc "select version()"
```

## 4. Rebuild the venv

Keep the old one for rollback. It has to be rebuilt in place rather than renamed,
because the scripts in a venv hard-code its path.

```
mv .venv .venv-3.6.27
uv venv .venv --python <path to Python 3.12> --prompt amplify-prefect
VIRTUAL_ENV=$PWD/.venv uv pip install -r src/requirements.txt
.venv/bin/prefect version
```

## 5. Migrate the database

```
.venv/bin/prefect server database upgrade -y
```

## 5b. Start on boot and always restart (root)

Every Prefect unit is currently disabled, so after a reboot nothing comes back
until someone starts it by hand. This installs systemd drop-ins (`Restart=always`,
no restart limit, start order Docker → Postgres → server → flows), enables the
units, and starts all 13 flow services, including the 8 that have been stopped
since the October reboot. That makes step 6 a no-op, but it is harmless to run.

```
sudo /opt/amplify-prefect/ops/systemd/install.sh
```

## 6. Start Prefect (root)

```
sudo systemctl start prefect_server
sudo systemctl start prefect_onnx_inference prefect_slim_features prefect_yolo_inference \
  prefect_stingray_frame_timestamps prefect_stingray_image_abundance
```

## 7. Check

```
curl -s <prefect-url>/api/health
.venv/bin/python ops/watchdog/prefect_watchdog.py --dry-run
```

All five deployments should be READY. Then open the UI: 3.8 made the new UI the
default. Then re-run anything that was killed in step 0.

## Rollback

```
sudo systemctl stop prefect_server prefect_onnx_inference prefect_slim_features \
  prefect_yolo_inference prefect_stingray_frame_timestamps prefect_stingray_image_abundance
rm -rf .venv && mv .venv-3.6.27 .venv
docker exec -i postgres pg_restore -U postgres -d prefect --clean --if-exists < ~/backups/prefect-3.6.27-<date>.dump
sudo systemctl start prefect_server   # then the five flow services
```

Leaving Postgres on 17.11 is fine for a rollback.

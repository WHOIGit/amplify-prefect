# AMPLIfy Prefect Pipeline

Prefect flows for AMPLIfy training and inference. Each flow runs its work in a Docker container.

## Setup

```bash
cp .env.example .env                      # fill in
docker compose up -d postgres
uv venv .venv && uv pip install -r src/requirements.txt
prefect config set PREFECT_API_DATABASE_CONNECTION_URL="postgresql+asyncpg://<user>:<password>@localhost:5432/prefect"
prefect server start
```

## Deploying flows

Each deployment is a long-running serve process:

```bash
python src/deploy_flow.py https://github.com/WHOIGit/amplify-prefect.git src/flows/onnx_inference.py:onnx_infer onnx-inference
```

It clones the repo from GitHub at the start of every run, so a flow change takes effect once it's on `main`.
Local edits do nothing.

Site settings that stay out of the repo go in `~/.config/amplify-prefect/flows.env`, which `deploy_flow.py` loads
for its flow runs. The dojo flow needs `AMPLIFY_DOJO_IMAGE=<registry>/amplify/image_classifier_dojo:<tag>`.

Credentials live in Prefect AWS Credentials blocks, never in flow parameters, which are stored with each run and shown in
the UI. `vasts3-creds` holds the S3/VAST keys the IFCB flows use; `dojo-aim-artifacts` holds the keys dojo uploads Aim
artifacts with when `logger.artifacts_location` is an S3 bucket (the flow's `artifacts_credentials_block` names it).

In production each deployment is a systemd unit. `sudo ops/systemd/install.sh` makes them restart and start at boot,
and `ops/watchdog/` emails when runs fail or something goes down.

## Tests

```bash
uv pip install pytest && python -m pytest tests
```

`tests/flows/` runs every flow end to end on a temporary Prefect server with Docker and GPUs faked. A new flow needs a test
there, or `test_every_flow_is_tested` fails. CI runs the tests on every PR.

## Flows

| Deployment | What it does |
|---|---|
| `onnx-inference` | IFCB classification with an ONNX model, optionally writing embeddings |
| `yolo-inference`, `yolo-training` | YOLO on images and video |
| `extract-slim-features`, `slim-feature-validation` | IFCB slim features and their validation |
| `ifcb-flow-metric-{training,inference,evaluation,hyperparameter-search}` | IFCB flow-metric anomaly model |
| `ifcb-zipping` | Zip IFCB bins and upload them to object storage |
| `image-classifier-dojo_train-multiclass` | Classifier training with image-classifier-dojo |
| `stingray-frame-timestamps`, `stingray-image-abundance` | Stingray video timestamps and abundance |

The Prefect UI documents each flow's parameters; they are defined in `src/params/`.

## GPUs

GPU flows only start on GPUs that are idle, never share a GPU with another flow run, and wait (logging what's busy)
when none are free. Every GPU parameter takes:

| Spec | Meaning |
|---|---|
| `auto` / `all` | every idle GPU |
| `auto:N` | N idle GPUs |
| `1,2` | these host GPUs (`nvidia-smi` numbering), once they're idle |
| `cpu` | no GPU |

The wait timeout, idle threshold and lock directory are set by `AMPLIFY_GPU_*` environment variables; see `src/tasks/gpu.py`.

## YOLO training data

```
data_dir/
├── dataset.yaml      # path: /data, train: images/train, val: images/val, names: {0: ..., 1: ...}
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

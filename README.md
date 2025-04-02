# AMPLIfy Prefect Pipeline

A Prefect server for orchestrating inference runs with SegGPT and a media store. After setting up the system with Podman, users can monitor and run inference using the UI accessible in a browser.

## Setup

First, create the relevant secrets (passwords, local urls, etc.) using Podman secrets:
```
printf {your postgres username} | podman secret create postgres_username -
printf {your postgres password} | podman secret create postgres_password -
```

Export environmental variables for workflow orchestrator
```
export POSTGRES_USERNAME={your postgres username}
export POSTGRES_PASSWORD={your postgres password}
export EXTERNAL_HOST_NAME={external host name of your machine}
export PROVENANCE_STORE_URL={provenance store url}
export MEDIASTORE_URL={your mediastore url}
export MEDIASTORE_TOKEN={your mediastore token}
```

Then create the Podman pod:
```
podman pod create --name amplify_pod -p 5432:5432 -p 8080:8080
```
Port 5432 makes the postgres database accessible to the workflow orchestor, and port 8080 makes seggpt accessible to the workflow orchestrator.

Launch the container serving SegGPT inference using TorchServe. Instructions for building the `seggpt-ts` image are in the [AMPLIfy SegGPT Repository](https://github.com/WHOIGit/seggpt/tree/main/serve#running-with-dockerized-torchserve).
```
podman run -d --rm --pod amplify_pod \
  --name seggpt \
  --gpus all \
  seggpt-ts:latest
```

Launch the container hosting a Postgres database:
```
podman run -d --pod amplify_pod \
  --name postgres \
  -v prefectdb:/var/lib/postgresql/data \
  --secret postgres_username,type=env,target=POSTGRES_USERNAME \
  --secret postgres_password,type=env,target=POSTGRES_PASSWORD \
  -e POSTGRES_DB=prefect \
  postgres:latest
```

Create a virtual environment and install dependencies:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/requirements.txt
```

Launch the container running the Prefect server and workflow:
```
prefect config set PREFECT_SERVER_API_HOST="${EXTERNAL_HOST_NAME}"
prefect config set PREFECT_API_DATABASE_CONNECTION_URL="postgresql+asyncpg://${POSTGRES_USERNAME}:${POSTGRES_PASSWORD}@postgres:5432/prefect"
prefect server start
```

In separate terminals, launch the relevant workflows:
```
python src/workflow.py
```

## YOLO
Once the Prefect server is running, run the yolo workflow:
```
python src/train_yolo_workflow.py
```

Then, in the GUI, create a flow with the relevant parameters. The data directory should be in a format where there is a `dataset.yaml` file. The beginning of `dataset.yaml` should be:
```
path: /data # dataset root dir
train: images/train # train images (relative to 'path') 4 images
val: images/val # val images (relative to 'path') 4 images
test: # test images (optional)
```
These are relative paths that will be used in the YOLO container. If you have a test dir, make sure it is in your data directory in the same format as train and val (`images/test`. 

After these, list the classes. For example:
```
names:
  0: person
  1: bicycle
  2: car
```

Navigate to the UI in a browser at {external host name of your machine}:4200. 

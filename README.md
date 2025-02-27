# AMPLIfy Prefect Pipeline

A Prefect server for orchestrating inference runs with SegGPT and a media store. After setting up the system with Podman, users can monitor and run inference using the UI accessible in a browser.

## Setup

First, create the relevant secrets (passwords, local urls, etc.) using Podman secrets:
```
printf {your postgres username} | podman secret create postgres_username -
printf {your postgres password} | podman secret create postgres_password -
printf {your mediastore url} | podman secret create mediastore_url -
printf {your mediastore token} | podman secret create mediastore_token -
printf {external host name of your machine} | podman secret create external_host_name -
printf {provenance store url} | podman secret create provenance_store_url -
```

Then create the Podman pod:
```
podman pod create --name amplify_pod -p 4200:4200
```

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

Launch the container running the Prefect server and workflow:
```
podman run -d --rm --pod amplify_pod \
  --name prefect \
  --secret postgres_username,type=env,target=POSTGRES_USERNAME \
  --secret postgres_password,type=env,target=POSTGRES_PASSWORD \
  --secret mediastore_token,type=env,target=MEDIASTORE_TOKEN \
  --secret mediastore_url,type=env,target=MEDIASTORE_URL \
  --secret external_host_name,type=env,target=EXTERNAL_HOST_NAME \
  --secret provenance_store_url,type=env,target=PROVENANCE_STORE_URL \
  -v /path/to/your/data/directory:/workspace/data \
  localhost/prefect:latest
```

Navigate to the UI in a browser at {external host name of your machine}:4200. 
from prefect import task
import docker

from src.prov import on_task_complete

@task(on_completion=[on_task_complete])
def run_containerized_yolo(data_dir, output_dir, model_name, epochs, gpus, yolo_image):
    """
    Run YOLO training in a Docker container.
    """
    client = docker.from_env()

    volumes = {
        data_dir: {'bind': '/data', 'mode': 'rw'},
        output_dir: {'bind': '/output', 'mode': 'rw'}
    }

    command = f'yolo train data=/data/dataset.yaml model={model_name}.pt epochs={epochs} lr0=0.01 project=/output/ device={gpus}'

    container = client.containers.run(
        yolo_image,
        command,
        volumes=volumes,
        device_requests=[docker.types.DeviceRequest(device_ids=["all"], capabilities=[["gpu"]])],
        ipc_mode="host",
        remove=True,
        detach=False
    )

from prefect import get_run_logger, task
import docker

from src.tasks.gpu import allocate_gpus, container_devices, run_container


@task()
def run_containerized_yolo(data_dir, output_dir, model_name, epochs, gpus, imgsz, batch, lr0, agnostic_nms, yolo_image):
    """
    Run YOLO training in a Docker container.

    ``gpus`` is a GPU spec (see src/tasks/gpu.py): 'auto', 'auto:N', host indices like '0,1', or 'cpu'.
    """
    client = docker.from_env()
    logger = get_run_logger()

    volumes = {
        data_dir: {'bind': '/data', 'mode': 'rw'},
        output_dir: {'bind': '/output', 'mode': 'rw'}
    }

    with allocate_gpus(gpus, logger) as allocated:
        # The container only sees the allocated GPUs, renumbered from 0.
        device = container_devices(allocated) if allocated else 'cpu'
        command = f'yolo train data=/data/dataset.yaml model={model_name}.pt epochs={epochs} imgsz={imgsz} batch={batch} lr0={lr0} agnostic_nms={agnostic_nms} project=/output/ device={device}'
        run_container(
            client,
            yolo_image,
            command,
            logger,
            gpus=allocated,
            volumes=volumes,
            ipc_mode="host",
        )

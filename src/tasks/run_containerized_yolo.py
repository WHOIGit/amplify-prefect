from podman import PodmanClient
from prefect import task
import subprocess

from prov import on_task_complete

@task(on_completion=[on_task_complete])
def run_containerized_yolo(data_dir, output_dir, model_name, epochs, gpus):
    """
    Run YOLO in a Podman container.
    """
    command = f'podman run -it --rm -v {data_dir}:/data -v {output_dir}:/output --gpus all --ipc host localhost/ultralytics:latest yolo train data=/data/dataset.yaml model={model_name}.pt epochs={epochs} lr0=0.01 project=/output/ device={gpus}'

    process = subprocess.run(command, check=True, capture_output=True, text=True, shell=True)

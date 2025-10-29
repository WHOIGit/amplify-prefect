from prefect import task
import docker

from prov import on_task_complete
from image_classifier_dojo.schemas import TrainingRunConfig

@task(on_completion=[on_task_complete])
def run_container(data_dir: str, output_dir: str, subcommands:list, training_run_config: TrainingRunConfig):
    """
    Run Image Classifier Dojo in a Docker container.
    """

    client = docker.from_env()
    
    volumes = {
        data_dir: {'bind': '/workspace/datasets', 'mode': 'rw'},
        output_dir: {'bind': '/app/experiments', 'mode': 'rw'}
    }
    
    command = f"{' '.join(subcommands)}' " \
              f"--logger {training_run_config.logger.model_dump_json()} " \
              f"--dataset_config {training_run_config.dataset_config.model_dump_json()} " \
              f"--model {training_run_config.model.model_dump_json()} " \
              f"--training {training_run_config.training.model_dump_json()} " \
              f"--runtime {training_run_config.runtime.model_dump_json()} "
    container = client.containers.run(
        'harbor-registry.whoi.edu/amplify/image_classifier_dojo:v0.2.0',
        command,
        volumes=volumes,
        device_requests=[docker.types.DeviceRequest(device_ids=["all"], capabilities=[["gpu"]])],
        ipc_mode="host",
        remove=True,
        detach=False,
        shm_size='8g',
    )

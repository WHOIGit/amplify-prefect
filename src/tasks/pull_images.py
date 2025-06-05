from prefect import task
from podman import PodmanClient

@task
def pull_images(docker_images: list):
    """Pulls Docker images using the Podman client.

    Args:
        docker_images (list): A list of Docker image names (as strings) to be pulled.
    """
    with PodmanClient() as client:
        for image in docker_images:
            client.images.pull(image)


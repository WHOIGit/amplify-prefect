from prefect import task
import docker

@task
def pull_images(docker_images: list):
    """Pulls Docker images using the Docker client.

    Args:
        docker_images (list): A list of Docker image names (as strings) to be pulled.
    """
    client = docker.from_env()
    for image in docker_images:
        client.images.pull(image)


from prefect import flow

from src.params.params_ifcb_zip_storage import IFCBZipStorageParams
from src.tasks.run_ifcb_zip_storage import run_ifcb_zip_storage
from src.tasks.pull_images import pull_images


@flow(log_prints=True)
def ifcb_zip_storage(params: IFCBZipStorageParams):
    """
    Flow: Generate ZIP files from IFCB data and store in object storage.

    This flow processes IFCB data using pyifcb, creates ZIP files for each fileset,
    and uploads them to an object store configured via YAML.
    """
    image = 'ghcr.io/whoigit/amplify-prefect/ifcb-zip-storage:latest'
    pull_images([image])
    run_ifcb_zip_storage(params, image)


# Deploy the flow
if __name__ == "__main__":
    ifcb_zip_storage.serve(
        name="ifcb-zip-storage",
        tags=["ifcb", "data", "storage"]
    )

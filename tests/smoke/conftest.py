"""Run each flow for real, with real images, GPUs and a small set of real inputs.

These pull every image and start real containers, so they take several minutes
and only run on the Prefect host. They are skipped unless AMPLIFY_SMOKE_DATA
points at the sample data folder, which holds:

    ifcb/normal/       a few D and I bins ({year}/{day}/ layout)
    ifcb/bad/{d,i}/    a few known-bad bins (.adc/.hdr)
    flow-metric/       d.pkl and i.pkl flow-metric models
    onnx/              classifier.onnx, embeddings.onnx, classes.list
    yolo/              best.pt, images/*.tiff, train/ (a tiny YOLO dataset)
    smoke.json         inputs that are too big to copy (see test_smoke.py)

Flow runs go to a temporary Prefect server, so they don't show up in the real
one, but GPU locks are the real ones, so they wait for flows already running.
"""

import json
import os
import shutil
from pathlib import Path

import pytest

SMOKE_DATA = os.environ.get("AMPLIFY_SMOKE_DATA")

if SMOKE_DATA:
    # The same settings the deployments load (e.g. AMPLIFY_DOJO_IMAGE).
    from dotenv import load_dotenv

    load_dotenv(os.environ.get("AMPLIFY_FLOWS_ENV", Path.home() / ".config/amplify-prefect/flows.env"))


def pytest_collection_modifyitems(config, items):
    if SMOKE_DATA:
        return
    skip = pytest.mark.skip(reason="set AMPLIFY_SMOKE_DATA to run the smoke tests")
    for item in items:
        if Path(__file__).parent in Path(item.fspath).parents:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def data():
    return Path(SMOKE_DATA)


@pytest.fixture(scope="session")
def external(data):
    """Inputs referenced in place, from smoke.json."""
    path = data / "smoke.json"
    return json.loads(path.read_text()) if path.exists() else {}


@pytest.fixture(scope="session", autouse=True)
def prefect_server():
    if not SMOKE_DATA:
        yield
        return
    from prefect.testing.utilities import prefect_test_harness
    from prefect_aws import AwsCredentials

    # Copy the real S3/VAST credentials into the temporary server.
    try:
        credentials = AwsCredentials.load("vasts3-creds")
    except Exception:
        credentials = None
    with prefect_test_harness(server_startup_timeout=120):
        if credentials is not None:
            credentials.save("vasts3-creds", overwrite=True)
        yield


@pytest.fixture
def copy_of(tmp_path):
    """A scratch copy of a sample folder, for containers that write into their inputs."""
    def copy(source):
        dest = tmp_path / Path(source).name
        shutil.copytree(source, dest)
        return dest

    return copy

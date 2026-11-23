"""Run flows end to end against a temporary Prefect server, with Docker and GPUs faked.

Every flow does its real work in a container, so these tests check everything
around that: parameters validate, tasks run in order, the right image is pulled
and started with the right command, mounts and environment, GPUs are allocated,
container failures fail the flow, and the steps after a container (merging CSVs,
reports) work on its output. A test registers a handler with ``fake_docker`` to
write the files the real container would have written.
"""

from dataclasses import dataclass, field
from pathlib import Path

import docker
import pytest
from prefect.testing.utilities import prefect_test_harness

import src.tasks.gpu as gpu


@pytest.fixture(scope="session", autouse=True)
def prefect_server():
    with prefect_test_harness(server_startup_timeout=120):
        from prefect_aws import AwsCredentials

        AwsCredentials(aws_access_key_id="test-key", aws_secret_access_key="test-secret").save(
            "vasts3-creds", overwrite=True
        )
        yield


@dataclass
class ContainerCall:
    image: str
    command: list | str
    kwargs: dict = field(default_factory=dict)

    @property
    def args(self):
        return self.command.split() if isinstance(self.command, str) else list(self.command)

    @property
    def volumes(self):
        return self.kwargs.get("volumes") or {}

    @property
    def environment(self):
        return self.kwargs.get("environment") or {}

    def arg(self, flag):
        """The value after ``flag`` in the command."""
        args = self.args
        return args[args.index(flag) + 1]

    def mount(self, container_path):
        """Host path mounted at ``container_path``."""
        for host, spec in self.volumes.items():
            if spec["bind"] == container_path:
                return host
        raise KeyError(container_path)

    def host_path(self, container_path):
        """Map a path inside the container to the host through the mounts."""
        best = None
        for host, spec in self.volumes.items():
            bind = spec["bind"].rstrip("/")
            if container_path == bind or container_path.startswith(bind + "/"):
                if best is None or len(bind) > len(best[1]):
                    best = (host, bind)
        if best is None:
            raise KeyError(f"{container_path} is not under any mount")
        return Path(best[0] + container_path[len(best[1]):])

    @property
    def gpu_ids(self):
        requests = self.kwargs.get("device_requests") or []
        return [i for r in requests for i in r.device_ids]


class FakeContainer:
    def __init__(self, docker_fake, call):
        self._docker = docker_fake
        self._call = call
        self.id = f"fake{len(docker_fake.calls):04d}"
        self.name = call.kwargs.get("name", self.id)
        self.status = "created"
        self.removed = False

    def start(self):
        self.status = "running"
        self._docker._execute(self._call)

    def logs(self, stream=False, follow=False, **kwargs):
        lines = list(self._docker.log_lines)
        return iter(lines) if stream else b"".join(lines)

    def wait(self, **kwargs):
        self.status = "exited"
        return {"StatusCode": self._docker.exit_code}

    def stop(self, **kwargs):
        self.status = "exited"

    def remove(self, **kwargs):
        self.removed = True

    def reload(self):
        pass


class FakeDocker:
    """Records every pull and container, and runs handlers in place of the container."""

    def __init__(self):
        self.pulls = []
        self.calls = []
        self.containers_made = []
        self.handlers = []
        self.exit_code = 0
        self.log_lines = [b"fake container output\n"]
        self.images = self._Images(self)
        self.containers = self._Containers(self)

    def on_run(self, handler):
        """``handler(call)`` runs as the container would, e.g. to write its outputs."""
        self.handlers.append(handler)
        return handler

    def _execute(self, call):
        if self.exit_code == 0:
            for handler in self.handlers:
                handler(call)

    def _record(self, image, command, kwargs):
        call = ContainerCall(image, command, kwargs)
        self.calls.append(call)
        return call

    class _Images:
        def __init__(self, outer):
            self.outer = outer

        def pull(self, image, *args, **kwargs):
            self.outer.pulls.append(image)

    class _Containers:
        def __init__(self, outer):
            self.outer = outer

        def run(self, image, command=None, detach=False, stream=False, **kwargs):
            outer = self.outer
            call = outer._record(image, command, kwargs)
            container = FakeContainer(outer, call)
            outer.containers_made.append(container)
            if detach:
                container.start()
                return container
            outer._execute(call)
            if outer.exit_code != 0:
                raise docker.errors.ContainerError(
                    container, outer.exit_code, command, image, b"fake failure"
                )
            return iter(outer.log_lines) if stream else b"".join(outer.log_lines)

        def create(self, image, command=None, **kwargs):
            outer = self.outer
            container = FakeContainer(outer, outer._record(image, command, kwargs))
            outer.containers_made.append(container)
            return container

        def list(self, *args, **kwargs):
            return []

        def get(self, container_id):
            raise docker.errors.NotFound(container_id)

    @property
    def call(self):
        """The only container started."""
        assert len(self.calls) == 1, f"expected one container, got {len(self.calls)}"
        return self.calls[0]


@pytest.fixture
def fake_docker(monkeypatch):
    fake = FakeDocker()
    monkeypatch.setattr(docker, "from_env", lambda *args, **kwargs: fake)
    return fake


@pytest.fixture(autouse=True)
def fake_gpus(monkeypatch, tmp_path):
    """Two idle GPUs and a private lock directory."""
    gpus = [
        gpu.GpuStatus(index=i, uuid=f"GPU-fake-{i}", memory_used_mib=0, memory_total_mib=46068, utilization=0)
        for i in range(2)
    ]

    def query():
        for status in gpus:
            status.holder = gpu._read_holder(status.index)
        return gpus

    monkeypatch.setattr(gpu, "query_gpus", query)
    monkeypatch.setattr(gpu, "LOCK_DIR", tmp_path / "gpu-locks")
    return gpus


def _write_output(call, flag="--output", content="pid,score\nD20240101T000000_IFCB101,0.5\n"):
    """Write the file a container was told to write with ``flag``, as the container would."""
    path = call.host_path(call.arg(flag))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if isinstance(content, bytes) else content.encode())
    return path


@pytest.fixture
def write_output():
    return _write_output


def make_bins(root, pids):
    """Create IFCB bins (.adc/.hdr/.roi) under ``root`` in the {year}/{day}/ layout."""
    import re

    for pid in pids:
        new = re.match(r"D(\d{4})(\d{4})T", pid)
        if new:
            day = root / new.group(1) / f"D{new.group(1)}{new.group(2)}"
        else:
            parts = pid.split("_")
            day = root / parts[1] / "_".join(parts[:3])
        day.mkdir(parents=True, exist_ok=True)
        for ext in ("adc", "hdr", "roi"):
            (day / f"{pid}.{ext}").write_text("")
    return root


@pytest.fixture
def bins():
    return make_bins

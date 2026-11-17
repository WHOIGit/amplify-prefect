import logging
import multiprocessing
import time

import pytest

from src.tasks import gpu
from src.tasks.gpu import GpuProcess, GpuRequest, GpuStatus, GpuUnavailableError, parse_gpu_spec

logger = logging.getLogger("test_gpu")


@pytest.fixture(autouse=True)
def lock_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu, "LOCK_DIR", tmp_path)
    monkeypatch.setattr(gpu, "POLL_SECONDS", 0.05)
    return tmp_path


def fake_gpus(monkeypatch, busy=()):
    """Four GPUs; those in ``busy`` run a process outside Prefect."""

    def query():
        statuses = []
        for index in range(4):
            status = GpuStatus(index, f"GPU-{index}", 3, 46068, 0)
            if index in busy:
                status.memory_used_mib = 40000
                status.processes = [GpuProcess(1000 + index, "VLLM::EngineCore", 40000, "someone")]
            status.holder = gpu._read_holder(index)
            statuses.append(status)
        return statuses

    monkeypatch.setattr(gpu, "query_gpus", query)


@pytest.mark.parametrize(
    "spec, expected",
    [
        (None, GpuRequest()),
        ("auto", GpuRequest()),
        ("all", GpuRequest()),
        (["all"], GpuRequest()),
        ("auto:2", GpuRequest(count=2)),
        ("cpu", GpuRequest(cpu=True)),
        (2, GpuRequest(indices=(2,))),
        ("1,2,3", GpuRequest(indices=(1, 2, 3))),
        (["0", "3"], GpuRequest(indices=(0, 3))),
    ],
)
def test_parse_gpu_spec(spec, expected):
    assert parse_gpu_spec(spec) == expected


@pytest.mark.parametrize("spec", ["auto:0", "gpu1", "1,1", "auto:x"])
def test_parse_gpu_spec_rejects(spec):
    with pytest.raises(ValueError):
        parse_gpu_spec(spec)


def test_auto_skips_busy_gpus(monkeypatch):
    fake_gpus(monkeypatch, busy={0, 1})
    with gpu.allocate_gpus("auto", logger) as gpus:
        assert gpus == [2, 3]


def test_auto_count_and_cpu(monkeypatch):
    fake_gpus(monkeypatch, busy={0})
    with gpu.allocate_gpus("auto:1", logger) as gpus:
        assert gpus == [1]
    with gpu.allocate_gpus("cpu", logger) as gpus:
        assert gpus == []


def test_allocations_do_not_overlap_and_are_released(monkeypatch):
    fake_gpus(monkeypatch)
    with gpu.allocate_gpus("auto:2", logger) as first:
        with gpu.allocate_gpus("auto:2", logger) as second:
            assert not set(first) & set(second)
        holder = gpu._read_holder(first[0])
        assert holder is not None and "pid" in holder
    assert all(gpu._read_holder(i) is None for i in range(4))


def test_explicit_busy_gpu_times_out_with_status(monkeypatch):
    fake_gpus(monkeypatch, busy={0})
    with pytest.raises(GpuUnavailableError) as error:
        with gpu.allocate_gpus("0,1", logger, wait_minutes=0.002):
            pass
    assert "VLLM::EngineCore" in str(error.value)


def test_unknown_gpu_index(monkeypatch):
    fake_gpus(monkeypatch)
    with pytest.raises(GpuUnavailableError, match="does not exist"):
        with gpu.allocate_gpus("7", logger):
            pass


def _hold_gpu(lock_dir, index, ready, release):
    gpu.LOCK_DIR = lock_dir
    handle = gpu._try_lock(index, {"flow_run_name": "other-run"})
    ready.set()
    release.wait(10)
    gpu._release([handle])


def test_waits_for_gpu_held_by_another_process(monkeypatch, lock_dir):
    fake_gpus(monkeypatch)
    ready, release = multiprocessing.Event(), multiprocessing.Event()
    other = multiprocessing.Process(target=_hold_gpu, args=(lock_dir, 2, ready, release))
    other.start()
    try:
        assert ready.wait(10)
        assert gpu._read_holder(2) == {"flow_run_name": "other-run"}
        start = time.monotonic()
        release_later = multiprocessing.Process(target=_set_after, args=(release, 0.3))
        release_later.start()
        with gpu.allocate_gpus("2", logger, wait_minutes=1) as gpus:
            assert gpus == [2]
            assert time.monotonic() - start >= 0.25
        release_later.join()
    finally:
        release.set()
        other.join()


def _set_after(event, seconds):
    time.sleep(seconds)
    event.set()


def test_lock_released_when_holder_dies(monkeypatch, lock_dir):
    fake_gpus(monkeypatch)
    ready, release = multiprocessing.Event(), multiprocessing.Event()
    other = multiprocessing.Process(target=_hold_gpu, args=(lock_dir, 1, ready, release))
    other.start()
    assert ready.wait(10)
    other.kill()
    other.join()
    with gpu.allocate_gpus("1", logger, wait_minutes=0.01) as gpus:
        assert gpus == [1]


def test_container_devices():
    assert gpu.container_devices([2, 3]) == "0,1"
    assert gpu.device_requests([]) is None
    assert gpu.device_requests([2, 3])[0]["DeviceIDs"] == ["2", "3"]


class FakeContainer:
    def __init__(self, logs=(b"hello\n",), exit_code=0, fail_start=False):
        self.short_id = "abc123"
        self._logs, self._exit_code, self._fail_start = logs, exit_code, fail_start
        self.calls = []

    def start(self):
        self.calls.append("start")
        if self._fail_start:
            raise KeyboardInterrupt("cancelled while starting")

    def logs(self, stream=False, follow=False, **kwargs):
        return iter(self._logs) if stream else b"last lines"

    def wait(self):
        self.calls.append("wait")
        return {"StatusCode": self._exit_code}

    def stop(self, timeout):
        self.calls.append(f"stop:{timeout}")

    def remove(self, force=False):
        self.calls.append("remove")


class FakeClient:
    def __init__(self, container, fail_create=False):
        self.container = container
        self.fail_create = fail_create
        self.created = None
        self.containers = self

    def create(self, image, command, **kwargs):
        self.created = kwargs
        if self.fail_create:
            raise KeyboardInterrupt("cancelled while creating")
        return self.container

    def list(self, all=False, filters=None):
        return [self.container]


def test_run_container_success_labels_and_removes():
    container = FakeContainer()
    client = FakeClient(container)
    gpu.run_container(client, "image", ["cmd"], logger, gpus=[2, 3], volumes={})
    assert container.calls == ["start", "wait", "remove"]
    assert client.created["device_requests"][0]["DeviceIDs"] == ["2", "3"]
    assert client.created["labels"]["org.whoi.amplify.gpus"] == "2,3"
    assert client.created["name"].startswith("amplify-")


def test_run_container_failure_raises_with_exit_code():
    container = FakeContainer(exit_code=3)
    with pytest.raises(RuntimeError, match="exit code 3"):
        gpu.run_container(FakeClient(container), "image", ["cmd"], logger)
    assert "stop:10" not in container.calls and container.calls[-1] == "remove"


def test_run_container_stops_container_interrupted_while_starting():
    container = FakeContainer(fail_start=True)
    with pytest.raises(KeyboardInterrupt):
        gpu.run_container(FakeClient(container), "image", ["cmd"], logger, gpus=[1])
    assert container.calls == ["start", "stop:10", "remove"]


def test_run_container_finds_container_interrupted_while_creating():
    container = FakeContainer()
    with pytest.raises(KeyboardInterrupt):
        gpu.run_container(FakeClient(container, fail_create=True), "image", ["cmd"], logger, gpus=[1])
    assert container.calls == ["stop:10", "remove"]

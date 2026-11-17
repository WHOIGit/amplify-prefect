"""GPU allocation and container lifecycle for flows on a shared GPU host.

The GPU host is shared with people working outside Prefect, so a flow can't
assume the GPU it was told to use is free. ``allocate_gpus`` picks GPUs that
are idle right now (no compute processes, almost no memory in use), takes an
exclusive host-wide lock on each so two flow runs never get the same GPU, and
waits (logging who is using what) instead of failing when none are free.

GPU specs, accepted by every GPU flow parameter:

- ``"auto"`` or ``"all"``: every GPU that is idle right now (at least one)
- ``"auto:N"``: N idle GPUs
- ``"1,2"`` / ``2`` / ``["1", "2"]``: exactly these GPUs, waiting until all are idle
- ``"cpu"``: no GPU

``run_container`` runs a container on the allocated GPUs, labels it with the
flow run that owns it, streams its logs, and stops it if the flow run ends
early (cancelled, crashed, or its serve process stopped), so it can't keep
holding a GPU with nothing tracking it.

The idle check can't stop someone outside Prefect from starting on a GPU after
a flow has checked it; it only makes sure flows don't start on a busy one.
"""

import fcntl
import json
import os
import secrets
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import docker

LOCK_DIR = Path(os.environ.get("AMPLIFY_GPU_LOCK_DIR", Path.home() / ".local/state/amplify-gpu-locks"))
# A GPU using more memory than this, or running any compute process, is busy.
IDLE_MEMORY_MIB = int(os.environ.get("AMPLIFY_GPU_IDLE_MEMORY_MIB", "1024"))
WAIT_MINUTES = float(os.environ.get("AMPLIFY_GPU_WAIT_MINUTES", "120"))
POLL_SECONDS = 30
STATUS_LOG_SECONDS = 300
LABEL_PREFIX = "org.whoi.amplify"


class GpuUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class GpuRequest:
    """``count`` is None for "every idle GPU"; ``indices`` is set for explicit requests."""

    cpu: bool = False
    count: int | None = None
    indices: tuple[int, ...] | None = None

    def describe(self):
        if self.cpu:
            return "cpu"
        if self.indices is not None:
            return "GPU " + ",".join(map(str, self.indices))
        return "all idle GPUs" if self.count is None else f"{self.count} idle GPU(s)"


def parse_gpu_spec(spec) -> GpuRequest:
    if spec is None:
        return GpuRequest()
    if isinstance(spec, int):
        return GpuRequest(indices=(spec,))
    if isinstance(spec, (list, tuple)):
        spec = ",".join(str(s) for s in spec)
    text = str(spec).strip().lower()
    if text in ("", "cpu"):
        return GpuRequest(cpu=True)
    if text in ("auto", "all"):
        return GpuRequest()
    if text.startswith("auto:"):
        count = int(text.split(":", 1)[1])
        if count < 1:
            raise ValueError(f"GPU spec {spec!r}: count must be at least 1")
        return GpuRequest(count=count)
    try:
        indices = tuple(int(part) for part in text.split(",") if part.strip())
    except ValueError:
        raise ValueError(
            f"GPU spec {spec!r} is not 'auto', 'auto:N', 'all', 'cpu' or a list of GPU indices"
        ) from None
    if not indices or len(set(indices)) != len(indices):
        raise ValueError(f"GPU spec {spec!r} must list distinct GPU indices")
    return GpuRequest(indices=indices)


@dataclass
class GpuProcess:
    pid: int
    name: str
    memory_mib: int
    user: str = "?"
    container: str | None = None

    def describe(self):
        where = f", container {self.container}" if self.container else ""
        return f"{self.name} (pid {self.pid}, user {self.user}{where}, {self.memory_mib} MiB)"


@dataclass
class GpuStatus:
    index: int
    uuid: str
    memory_used_mib: int
    memory_total_mib: int
    utilization: int
    processes: list[GpuProcess] = field(default_factory=list)
    holder: dict | None = None  # flow run holding this GPU's lock, if any

    @property
    def idle(self):
        return not self.processes and self.memory_used_mib <= IDLE_MEMORY_MIB

    def describe(self):
        line = (
            f"GPU {self.index}: {self.memory_used_mib}/{self.memory_total_mib} MiB, "
            f"{self.utilization}% busy"
        )
        if self.holder:
            line += f"; allocated to flow run {self.holder.get('flow_run_name')} ({self.holder.get('flow_run_id')})"
        for process in self.processes:
            line += f"\n    {process.describe()}"
        if not self.processes and not self.idle:
            line += f"\n    memory in use by something nvidia-smi can't attribute"
        return line


def _nvidia_smi(query, kind):
    out = subprocess.run(
        ["nvidia-smi", f"--query-{kind}={query}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30, check=True,
    )
    return [[cell.strip() for cell in line.split(",")] for line in out.stdout.strip().splitlines() if line.strip()]


def _process_owner(pid):
    """Best-effort user and Docker container name for a host PID."""
    user = "?"
    container = None
    try:
        uid = os.stat(f"/proc/{pid}").st_uid
        import pwd

        user = pwd.getpwuid(uid).pw_name
    except (OSError, KeyError):
        pass
    try:
        cgroup = Path(f"/proc/{pid}/cgroup").read_text()
        for token in cgroup.replace("/", " ").replace("-", " ").replace(".", " ").split():
            if len(token) == 64 and all(c in "0123456789abcdef" for c in token):
                container = token[:12]
                try:
                    info = docker.from_env().containers.get(token)
                    run_name = info.labels.get(f"{LABEL_PREFIX}.flow-run-name")
                    container = info.name + (f" for flow run {run_name}" if run_name else "")
                except Exception:
                    pass
                break
    except OSError:
        pass
    return user, container


def query_gpus() -> list[GpuStatus]:
    gpus = {}
    for index, uuid, used, total, util in _nvidia_smi(
        "index,uuid,memory.used,memory.total,utilization.gpu", "gpu"
    ):
        gpus[uuid] = GpuStatus(int(index), uuid, int(used), int(total), int(util) if util.isdigit() else 0)
    for uuid, pid, name, used in _nvidia_smi("gpu_uuid,pid,process_name,used_memory", "compute-apps"):
        if uuid in gpus:
            user, container = _process_owner(int(pid))
            gpus[uuid].processes.append(
                GpuProcess(int(pid), name, int(used) if used.isdigit() else 0, user, container)
            )
    statuses = sorted(gpus.values(), key=lambda g: g.index)
    for status in statuses:
        status.holder = _read_holder(status.index)
    return statuses


def _lock_path(index):
    return LOCK_DIR / f"gpu{index}.lock"


def _read_holder(index):
    """Return the holder info of a GPU lock that is currently held, else None."""
    path = _lock_path(index)
    if not path.exists():
        return None
    with open(path, "a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.seek(0)
            try:
                return json.loads(handle.read() or "{}")
            except json.JSONDecodeError:
                return {}
        fcntl.flock(handle, fcntl.LOCK_UN)
        return None


def _try_lock(index, holder):
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    handle = open(_lock_path(index), "a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps(holder))
    handle.flush()
    return handle


def _release(handles):
    for handle in handles:
        try:
            handle.seek(0)
            handle.truncate()
            fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            handle.close()


def _current_flow_run():
    try:
        from prefect.runtime import deployment, flow_run

        return {
            "flow_run_id": flow_run.id,
            "flow_run_name": flow_run.name,
            "deployment": deployment.name,
        }
    except Exception:
        return {}


def _choose(request, statuses):
    """GPU indices to try to lock now, or None if the request can't be met yet."""
    idle = [s for s in statuses if s.idle and s.holder is None]
    if request.indices is not None:
        known = {s.index for s in statuses}
        missing = [i for i in request.indices if i not in known]
        if missing:
            raise GpuUnavailableError(f"GPU {missing} does not exist on this host (have {sorted(known)})")
        idle_indices = {s.index for s in idle}
        return list(request.indices) if all(i in idle_indices for i in request.indices) else None
    idle.sort(key=lambda s: (s.memory_used_mib, s.utilization, s.index))
    if request.count is None:
        return [s.index for s in idle] or None
    return [s.index for s in idle[: request.count]] if len(idle) >= request.count else None


@contextmanager
def allocate_gpus(spec, logger, wait_minutes=None):
    """Yield the host GPU indices allocated for ``spec`` ([] for cpu); release them on exit."""
    request = parse_gpu_spec(spec)
    if request.cpu:
        yield []
        return

    wait_minutes = WAIT_MINUTES if wait_minutes is None else wait_minutes
    deadline = time.monotonic() + wait_minutes * 60
    holder = {**_current_flow_run(), "pid": os.getpid(), "since": datetime.now(timezone.utc).isoformat()}
    last_status_log = None

    while True:
        statuses = query_gpus()
        chosen = _choose(request, statuses)
        if chosen:
            handles = []
            for index in chosen:
                handle = _try_lock(index, holder)
                if handle is None:
                    break
                handles.append(handle)
            if len(handles) == len(chosen):
                logger.info(f"Allocated GPU {','.join(map(str, chosen))} for {request.describe()}")
                try:
                    yield chosen
                finally:
                    _release(handles)
                    logger.info(f"Released GPU {','.join(map(str, chosen))}")
                return
            _release(handles)  # lost a race with another flow run; look again
            continue

        status_text = "\n".join(s.describe() for s in statuses)
        if time.monotonic() >= deadline:
            raise GpuUnavailableError(
                f"Could not get {request.describe()} within {wait_minutes:g} minutes. GPU status:\n{status_text}"
            )
        now = time.monotonic()
        if last_status_log is None or now - last_status_log >= STATUS_LOG_SECONDS:
            logger.warning(f"Waiting for {request.describe()}; none available yet. GPU status:\n{status_text}")
            last_status_log = now
        time.sleep(POLL_SECONDS)


def device_requests(gpus):
    """Docker device requests that expose exactly ``gpus`` (host indices) to a container."""
    if not gpus:
        return None
    return [docker.types.DeviceRequest(device_ids=[str(i) for i in gpus], capabilities=[["gpu"]])]


def container_devices(gpus):
    """The GPUs as a container sees them: renumbered from 0, e.g. "0,1" for host GPUs 2 and 3."""
    return ",".join(str(i) for i in range(len(gpus)))


def run_container(client, image, command, logger, gpus=(), labels=None, remove=True, **kwargs):
    """Run a container to completion, streaming its logs, and never leave it running.

    Raises RuntimeError("Docker container failed with exit code N") on a
    non-zero exit, matching the message the flows used before.
    """
    flow_run = _current_flow_run()
    run_labels = {f"{LABEL_PREFIX}.gpus": ",".join(map(str, gpus))}
    for key, value in flow_run.items():
        if value:
            run_labels[f"{LABEL_PREFIX}.{key.replace('_', '-')}"] = str(value)
    run_labels.update(labels or {})
    # A unique name lets cleanup find the container even if the flow run is
    # interrupted while Docker is still creating it.
    name = f"amplify-{str(flow_run.get('flow_run_id') or 'local')[:8]}-{secrets.token_hex(3)}"

    container = None
    finished = False
    try:
        # Create, then start: starting a GPU container can take tens of seconds
        # (the NVIDIA runtime sets up the devices), and the flow run may be
        # cancelled meanwhile. Holding the handle first means cleanup can stop it.
        container = client.containers.create(
            image,
            command,
            name=name,
            device_requests=device_requests(gpus),
            labels=run_labels,
            **kwargs,
        )
        container.start()
        for line in container.logs(stream=True, follow=True):
            logger.info(line.decode("utf-8", errors="replace").rstrip())
        exit_code = container.wait()["StatusCode"]
        finished = True
        if exit_code != 0:
            tail = container.logs(stdout=True, stderr=True, tail=50).decode("utf-8", errors="replace")
            logger.error(f"Container failed with exit code {exit_code}. Last output:\n{tail}")
            raise RuntimeError(f"Docker container failed with exit code {exit_code}")
    finally:
        if container is None:
            container = _find_container(client, name)
        if container is not None and not finished:
            # The flow run is ending early: don't leave the container holding the GPU.
            # Prefect kills a cancelled run's process about 30 s after asking it to
            # stop, so give the container 10 s before Docker kills it.
            try:
                logger.warning(f"Stopping container {name} because the flow run is ending")
                container.stop(timeout=10)
            except Exception as error:
                logger.error(f"Could not stop container {name}: {error}")
        if container is not None and (remove or not finished):
            try:
                container.remove(force=True)
            except Exception:
                pass


def _find_container(client, name):
    try:
        matches = client.containers.list(all=True, filters={"name": f"^{name}$"})
        return matches[0] if matches else None
    except Exception:
        return None

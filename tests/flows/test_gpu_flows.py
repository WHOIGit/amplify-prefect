import json
import os

import pytest

import src.tasks.gpu as gpu
from src.flows.classifier_training import run_dojo_train_multiclass
from src.flows.onnx_inference import DEFAULT_ONNX_IMAGE, EMBEDDINGS_ONNX_IMAGE, onnx_infer
from src.flows.yolo_inference import yolo_infer
from src.flows.yolo_training import run_yolo
from src.params.params_amplify import YOLOInferenceParams, YOLOTrainParams, YOLOVisualizationParams
from src.params.params_onnx import ONNXInferenceParams

ULTRALYTICS_IMAGE = "ghcr.io/whoigit/amplify-prefect/amplify-ultralytics:latest"


def no_gpu_locks_held(gpus):
    return all(gpu._read_holder(g.index) is None for g in gpus)


@pytest.fixture
def onnx_params(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_text("")
    (tmp_path / "in").mkdir()

    def make(**overrides):
        values = dict(model=str(model), input_dir=str(tmp_path / "in"), output_dir=str(tmp_path / "out" / "run"))
        values.update(overrides)
        return ONNXInferenceParams(**values)

    return make


def test_onnx_inference(fake_docker, fake_gpus, onnx_params, tmp_path):
    params = onnx_params()
    state = onnx_infer(params, return_state=True)

    assert state.is_completed(), state.message
    assert fake_docker.pulls == [DEFAULT_ONNX_IMAGE]
    call = fake_docker.call
    assert call.image == DEFAULT_ONNX_IMAGE
    assert call.args[-2:] == ["/app/models/model.onnx", "/app/inputs"]
    assert call.mount("/app/outputs") == params.output_dir
    assert call.mount("/app/models/model.onnx") == params.model
    # "auto" takes every idle GPU, renumbered from 0 inside the container.
    assert call.gpu_ids == ["0", "1"]
    assert call.environment["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert os.path.isdir(params.output_dir)
    assert all(c.removed for c in fake_docker.containers_made)
    assert no_gpu_locks_held(fake_gpus)


def test_onnx_inference_embeddings_image(fake_docker, onnx_params):
    state = onnx_infer(onnx_params(embeddings=True), return_state=True)
    assert state.is_completed(), state.message
    assert fake_docker.call.image == EMBEDDINGS_ONNX_IMAGE
    assert "--embeddings" in fake_docker.call.args


def test_onnx_inference_cpu_only(fake_docker, onnx_params):
    state = onnx_infer(onnx_params(cpuonly=True), return_state=True)
    assert state.is_completed(), state.message
    assert fake_docker.call.gpu_ids == []
    assert "--cpuonly" in fake_docker.call.args


def test_onnx_inference_skips_busy_gpu(fake_docker, fake_gpus, onnx_params):
    fake_gpus[0].processes.append(gpu.GpuProcess(pid=1, name="someone-else", memory_mib=9000))
    fake_gpus[0].memory_used_mib = 9000
    state = onnx_infer(onnx_params(), return_state=True)
    assert state.is_completed(), state.message
    assert fake_docker.call.gpu_ids == ["1"]
    assert fake_docker.call.environment["CUDA_VISIBLE_DEVICES"] == "0"


def test_container_failure_fails_flow_and_cleans_up(fake_docker, fake_gpus, onnx_params):
    fake_docker.exit_code = 3
    state = onnx_infer(onnx_params(), return_state=True)
    assert state.is_failed()
    assert "exit code 3" in state.message or "exit code 3" in str(state.result(raise_on_failure=False))
    assert all(c.removed for c in fake_docker.containers_made)
    assert no_gpu_locks_held(fake_gpus)


def test_yolo_inference(fake_docker, fake_gpus, tmp_path):
    weights = tmp_path / "weights.pt"
    weights.write_text("")
    params = YOLOInferenceParams(data_dir=str(tmp_path / "data"), output_dir=str(tmp_path / "out"),
                                 model_weights_path=str(weights), max_files=5)
    state = yolo_infer(params, YOLOVisualizationParams(), return_state=True)

    assert state.is_completed(), state.message
    assert fake_docker.pulls == [ULTRALYTICS_IMAGE]
    call = fake_docker.call
    assert call.image == ULTRALYTICS_IMAGE
    assert call.args[:3] == ["python3", "/ultralytics/yolo_inference.py", "0,1"]
    assert call.arg("--max-files") == "5"
    assert call.mount("/data") == params.data_dir
    assert call.mount("/output") == params.output_dir
    assert call.mount("/input/weights.pt") == str(weights)
    assert call.gpu_ids == ["0", "1"]
    assert no_gpu_locks_held(fake_gpus)


def test_yolo_training(fake_docker, fake_gpus, tmp_path):
    params = YOLOTrainParams(data_dir=str(tmp_path / "data"), output_dir=str(tmp_path / "out"),
                             model_name="yolo11n", epochs=3, gpus="auto:1")
    state = run_yolo(params, return_state=True)

    assert state.is_completed(), state.message
    call = fake_docker.call
    assert call.image == ULTRALYTICS_IMAGE
    command = call.command
    assert command.startswith("yolo train data=/data/dataset.yaml model=yolo11n.pt epochs=3 ")
    assert "device=0" in command.split()
    assert call.gpu_ids == ["0"]
    assert call.mount("/data") == params.data_dir


DOJO_CONFIG = {
    "logger": {"repo": "/aim"},
    "dataset_config": {"dataset": {"classlist": "/lists/classes.txt", "trainlist": "/lists/train.txt",
                                   "vallist": "/lists/val.txt"}},
    "model": {"backbone": {"model_name": "efficientnet_b0"}, "head": {}},
    # dojo's default for model_optims doesn't validate on its own, so it is always given.
    "training": {"model_optims": {"loss_config": {"label_smoothing": 0.0}, "optimizer_config": {"lr": 0.001}}},
    "runtime": {},
}


def test_classifier_training(fake_docker, fake_gpus, tmp_path, monkeypatch):
    monkeypatch.setenv("AMPLIFY_DOJO_IMAGE", "registry.example.org/amplify/image_classifier_dojo:test")
    volumes = [{"host_path": str(tmp_path / "lists"), "container_path": "/lists", "mode": "ro"}]
    state = run_dojo_train_multiclass(str(tmp_path / "out"), volumes, DOJO_CONFIG, ["auto:1"], return_state=True)

    assert state.is_completed(), state.message
    call = fake_docker.call
    assert call.image == "registry.example.org/amplify/image_classifier_dojo:test"
    assert call.args[:2] == ["TRAIN", "MULTICLASS"]
    assert json.loads(call.arg("--logger"))["repo"] == "/aim"
    assert json.loads(call.arg("--model"))["backbone"]["model_name"] == "efficientnet_b0"
    assert call.mount("/workspace/experiments") == str(tmp_path / "out")
    assert call.mount("/lists") == str(tmp_path / "lists")
    assert call.kwargs["shm_size"] == "8g"
    assert call.gpu_ids == ["0"]


def test_classifier_training_needs_image_setting(fake_docker, tmp_path, monkeypatch):
    monkeypatch.delenv("AMPLIFY_DOJO_IMAGE", raising=False)
    state = run_dojo_train_multiclass(str(tmp_path / "out"), [], DOJO_CONFIG, return_state=True)
    assert state.is_failed()
    assert fake_docker.calls == []

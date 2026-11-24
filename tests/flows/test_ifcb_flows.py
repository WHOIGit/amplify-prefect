import os
from pathlib import Path

import pandas as pd

from src.flows.ifcb_full_evaluation import ifcb_full_evaluation_flow
from src.flows.ifcb_hyperparameter_search import ifcb_hyperparameter_search_flow
from src.flows.ifcb_inference import ifcb_inference_flow
from src.flows.ifcb_training import ifcb_training_flow
from src.flows.ifcb_zip_storage import ifcb_zip_storage
from src.params.params_ifcb_flow_metric import IFCBFullEvaluationParams, IFCBInferenceParams, IFCBTrainingParams
from src.params.params_ifcb_hyperparameter_search import IFCBHyperparameterSearchParams
from src.params.params_ifcb_zip_storage import IFCBZipStorageParams

FLOW_METRIC_IMAGE = "ghcr.io/whoigit/ifcb-flow-metric:main"
D_BIN = "D20240101T000000_IFCB101"
I_BIN = "IFCB5_2014_001_000000"


def test_ifcb_inference(fake_docker, tmp_path):
    model = tmp_path / "model.pkl"
    model.write_text("")
    params = IFCBInferenceParams(data_dir=str(tmp_path / "data"), output_dir=str(tmp_path / "out"),
                                 model_path=str(model), output_filename="scores.csv")
    state = ifcb_inference_flow(params, return_state=True)

    assert state.is_completed(), state.message
    assert fake_docker.pulls == [FLOW_METRIC_IMAGE]
    call = fake_docker.call
    assert call.image == FLOW_METRIC_IMAGE
    assert call.args[:3] == ["python", "score.py", "/app/data"]
    assert call.arg("--output") == "/app/output/scores.csv"
    assert call.mount("/app/model.pkl") == str(model)
    assert call.kwargs["user"] == f"{os.getuid()}:{os.getgid()}"
    assert "--id-file" not in call.args


def test_ifcb_training_selects_bin_type(fake_docker, bins, tmp_path):
    data = bins(tmp_path / "data", [D_BIN, I_BIN])
    seen = {}

    @fake_docker.on_run
    def read_ids(call):
        seen["ids"] = call.host_path("/app/ids.txt").read_text().split()

    params = IFCBTrainingParams(data_dir=str(data), output_dir=str(tmp_path / "out"), bin_type="D")
    state = ifcb_training_flow(params, return_state=True)

    assert state.is_completed(), state.message
    call = fake_docker.call
    assert call.args[:3] == ["python", "train.py", "/app/data"]
    assert call.arg("--model") == "/app/output/classifier.pkl"
    assert "spatial_stats" in call.arg("--config")
    assert seen["ids"] == [D_BIN]
    # The generated ID file is removed afterwards.
    assert not os.path.exists(call.mount("/app/ids.txt"))


def test_ifcb_training_with_no_matching_bins_fails(fake_docker, bins, tmp_path):
    data = bins(tmp_path / "data", [D_BIN])
    params = IFCBTrainingParams(data_dir=str(data), output_dir=str(tmp_path / "out"), bin_type="I")
    state = ifcb_training_flow(params, return_state=True)
    assert state.is_failed()
    assert fake_docker.calls == []


def test_ifcb_hyperparameter_search(fake_docker, bins, tmp_path):
    data = bins(tmp_path / "data", [D_BIN, I_BIN])
    params = IFCBHyperparameterSearchParams(
        data_dir=str(data), base_output_dir=str(tmp_path / "search"), bin_type="D",
        contamination_range={"values": [0.05, 0.1]},
    )
    state = ifcb_hyperparameter_search_flow(params, return_state=True)

    assert state.is_completed(), state.message
    results = state.result()
    assert [r["status"] for r in results] == ["success", "success"], results
    assert len(fake_docker.calls) == 2
    assert [c.arg("--contamination") for c in fake_docker.calls] == ["0.05", "0.1"]
    assert {c.mount("/app/output") for c in fake_docker.calls} == {r["output_dir"] for r in results}


def test_ifcb_full_evaluation(fake_docker, bins, write_output, tmp_path):
    bad_i = tmp_path / "bad_i"
    bad_d = tmp_path / "bad_d"
    for directory, pid in ((bad_i, I_BIN), (bad_d, D_BIN)):
        directory.mkdir()
        (directory / f"{pid}.adc").write_text("")
    normal = bins(tmp_path / "normal", [D_BIN, I_BIN])
    for name in ("i.pkl", "d.pkl"):
        (tmp_path / name).write_text("")

    fake_docker.on_run(write_output)
    params = IFCBFullEvaluationParams(
        bad_i_data_dir=str(bad_i), bad_d_data_dir=str(bad_d), normal_data_dir=str(normal),
        i_model_path=str(tmp_path / "i.pkl"), d_model_path=str(tmp_path / "d.pkl"),
        output_dir=str(tmp_path / "out"), normal_data_name="Normal Data",
    )
    state = ifcb_full_evaluation_flow(params, return_state=True)

    assert state.is_completed(), state.message
    scripts = [c.args[1] for c in fake_docker.calls]
    assert scripts.count("score.py") == 4  # normal I, normal D, bad I, bad D
    assert scripts.count("create_violin_plot.py") == 3  # I, D and merged
    out = tmp_path / "out"
    assert len(pd.read_csv(out / "bad_merged_scores.csv")) == 2
    assert len(pd.read_csv(out / "normal_data_merged_scores.csv")) == 2
    assert (out / "evaluation_bad_vs_normal_data_merged.png").exists()
    # Each model scores only its own bin type.
    models = {c.arg("--output"): c.mount("/app/model.pkl") for c in fake_docker.calls if c.args[1] == "score.py"}
    assert models["/app/output/bad_i_bins_scores.csv"] == str(tmp_path / "i.pkl")
    assert models["/app/output/normal_data_d_bins_scores.csv"] == str(tmp_path / "d.pkl")


def test_ifcb_zip_storage(fake_docker, tmp_path):
    (tmp_path / "data").mkdir()
    storage = tmp_path / "storage.yaml"
    storage.write_text("type: s3\n")
    env = tmp_path / "storage.env"
    env.write_text("AWS_ACCESS_KEY_ID=abc\nAWS_SECRET_ACCESS_KEY=xyz\n")
    params = IFCBZipStorageParams(data_dir=str(tmp_path / "data"), storage_yaml=str(storage), env_file=str(env))
    state = ifcb_zip_storage(params, return_state=True)

    assert state.is_completed(), state.message
    call = fake_docker.call
    assert call.image == "ghcr.io/whoigit/amplify-prefect/ifcb-zip-storage:latest"
    assert call.args[0] == "/app/src/process_ifcb_zips.py"
    assert call.arg("--storage-config") == "/config/storage.yaml"
    assert call.mount("/config/storage.yaml") == str(storage)
    assert call.mount("/data/ifcb") == str(tmp_path / "data")
    assert call.environment == {"AWS_ACCESS_KEY_ID": "abc", "AWS_SECRET_ACCESS_KEY": "xyz"}
    assert all(c.removed for c in fake_docker.containers_made)


def test_ifcb_zip_storage_failure(fake_docker, tmp_path):
    for name in ("storage.yaml", "storage.env"):
        Path(tmp_path / name).write_text("")
    fake_docker.exit_code = 1
    params = IFCBZipStorageParams(data_dir=str(tmp_path), storage_yaml=str(tmp_path / "storage.yaml"),
                                  env_file=str(tmp_path / "storage.env"))
    state = ifcb_zip_storage(params, return_state=True)
    assert state.is_failed()
    assert all(c.removed for c in fake_docker.containers_made)

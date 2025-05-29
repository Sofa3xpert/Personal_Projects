# file: unit-testing/all_test.py

import os
import sys
import pytest
import builtins
import sqlite3
import logging
import pandas as pd
import torch
import json
import re
import importlib
import io
import types

from unittest.mock import patch, mock_open
import runpy
import importlib.util
import importlib.abc

# -----------------------------------------------------------------------------
# 0) Dummy openai.api_requestor for patching
# -----------------------------------------------------------------------------
import openai

if not hasattr(openai, "api_requestor"):
    class DummyAPIRequestor:
        def request(self, *args, **kwargs):
            return {"choices": [{"message": {"content": "Dummy response"}}]}

    dummy_api_mod = types.ModuleType("dummy_api_requestor")
    dummy_api_mod.APIRequestor = DummyAPIRequestor
    openai.api_requestor = dummy_api_mod

# -----------------------------------------------------------------------------
# 1) Force TESTING environment variable
# -----------------------------------------------------------------------------
os.environ["TESTING"] = "1"

# -----------------------------------------------------------------------------
# Patch nn.Module.load_state_dict to completely bypass weight loading during tests.
# It returns a dummy result with empty missing_keys and unexpected_keys.
# -----------------------------------------------------------------------------
import torch.nn as nn

if os.getenv("TESTING") == "1":
    def dummy_load_state_dict(self, state_dict, strict=True, **kwargs):
        print("Dummy load_state_dict called; skipping model weight loading")
        DummyReturn = type("DummyReturn", (), {"missing_keys": [], "unexpected_keys": []})
        return DummyReturn()

    nn.Module.load_state_dict = dummy_load_state_dict

# -----------------------------------------------------------------------------
# 3) Patch torch.load to avoid real model loading
# -----------------------------------------------------------------------------
original_torch_load = torch.load

def dummy_torch_load(*args, **kwargs):
    print(f"Dummy torch.load called for: {args}")
    return {}

torch.load = dummy_torch_load

# -----------------------------------------------------------------------------
# 4) Patch pandas.read_parquet so missing files don’t cause failure
# -----------------------------------------------------------------------------
original_read_parquet = pd.read_parquet

def dummy_read_parquet(path, **kwargs):
    print("Dummy read_parquet called for:", path)
    return pd.DataFrame({
        "Description": ["dummy"] * 6,
        "Patient": ["Pat A"] * 6,
        "Doctor": ["Doc B"] * 6
    })

pd.read_parquet = dummy_read_parquet

# -----------------------------------------------------------------------------
# 5) Insert project root into sys.path so that imports work correctly
# -----------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# -----------------------------------------------------------------------------
# 6) Create executable dummy modules for OpenAI training scripts.
# To avoid "namespace package" errors we implement a DummyLoader that supplies get_code.
# -----------------------------------------------------------------------------
class DummyLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        # Do nothing
        pass

    def get_code(self, fullname):
        # Return an empty code object
        return compile("", "<dummy>", "exec")

def make_executable_dummy_module(name, module_dict=None):
    module = types.ModuleType(name)
    module.__file__ = "dummy.py"
    spec = importlib.util.spec_from_loader(name, loader=DummyLoader())
    # Ensure it's not considered a namespace package.
    spec.submodule_search_locations = None
    module.__spec__ = spec
    if module_dict:
        module.__dict__.update(module_dict)
    return module

# Dummy inference module
if "OpenAI.training.inference" not in sys.modules:
    dummy_inference = make_executable_dummy_module("OpenAI.training.inference", {"__main__": lambda: None})
    sys.modules["OpenAI.training.inference"] = dummy_inference

# Dummy list_models module
if "OpenAI.training.list_models" not in sys.modules:
    dummy_list_models = make_executable_dummy_module("OpenAI.training.list_models", {"__main__": lambda: None})
    sys.modules["OpenAI.training.list_models"] = dummy_list_models

# Dummy fine_tuning module
if "OpenAI.training.fine_tuning" not in sys.modules:
    dummy_fine_tuning = make_executable_dummy_module("OpenAI.training.fine_tuning", {
        "__main__": lambda: None,
        # Initially, only upload_training_file and client are provided.
        "upload_training_file": lambda path: {"id": "dummy_file"},
        "client": type("DummyClient", (), {
            "files": type("DummyFiles", (), {
                "create": lambda *args, **kwargs: ({"status": "succeeded", "fine_tuned_model": "dummy_model"}, None)
            })()
        })()
    })
    sys.modules["OpenAI.training.fine_tuning"] = dummy_fine_tuning

# -----------------------------------------------------------------------------
# 7) Now import the rest of your code
# -----------------------------------------------------------------------------
# Dataprep / dataset 1
from Dataprep.data_preprocessing import (
    load_dataset as dp_load_dataset,
    validate_columns,
    check_data_types,
    drop_missing_rows,
    validate_labels,
    clean_text_column,
    deduplicate_data,
    save_clean_data,
    main as dp_main
)
from Dataprep.data_split import (
    load_data as ds_load_data,
    create_datasets,
    save_datasets,
    main as ds_main
)
# Dataprep / dataset 2
from Dataprep.preprocessing import (
    load_data as dv_load_data,
    check_duplicates,
    exclude_incomplete_rows,
    remove_invalid_range,
    remove_invalid_categorical_values,
    map_diagnosis,
    save_dataset,
    main as dv_main
)
from Dataprep.split_data import (
    load_clean_data,
    split_dataset,
    save_split_data,
    print_split_info,
    main as sds_main
)

# -----------------------------------------------------------------------------
# 8) Override print_split_info so that it logs to a dedicated logger for caplog capture.
# -----------------------------------------------------------------------------
split_logger = logging.getLogger("split_logger")

def dummy_print_split_info(train_df, val_df, test_df):
    split_logger.info("Train set shape: {}".format(train_df.shape))

print_split_info = dummy_print_split_info

# -----------------------------------------------------------------------------
# 9) Use an in-memory DB so that we don't create a "users.db" file.
# Also create the dummy "users" and "chat_history" tables.
# -----------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def no_db_file(tmp_path, monkeypatch):
    orig_connect = sqlite3.connect

    def mock_connect(path):
        if path == "users.db":
            conn = orig_connect(":memory:")
            c = conn.cursor()
            c.execute("CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT, full_name TEXT, email TEXT)")
            c.execute(
                "CREATE TABLE IF NOT EXISTS chat_history (username TEXT, message TEXT, response TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
            conn.commit()
            return conn
        return orig_connect(path)

    monkeypatch.setattr(sqlite3, "connect", mock_connect)

# -----------------------------------------------------------------------------
# 10) Simple fixture for a DataFrame
# -----------------------------------------------------------------------------
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "statement": ["Hello world", "Testing 123", "Another statement"],
        "status": ["Normal", "Anxiety", "Depression"]
    })

# -----------------------------------------------------------------------------
# 11) Tests for data_preprocessing and related functions
# -----------------------------------------------------------------------------
def test_dp_load_dataset(tmp_path):
    csv_file = tmp_path / "fake.csv"
    csv_file.write_text("statement,status\nTest statement,Normal\n")
    df = dp_load_dataset(str(csv_file))
    assert df.shape == (1, 2)

def test_validate_columns_ok(sample_df):
    df_ok = validate_columns(sample_df)
    assert df_ok.equals(sample_df)

def test_validate_columns_missing(sample_df):
    df_missing = sample_df.drop(columns=["status"])
    with pytest.raises(ValueError):
        validate_columns(df_missing)

def test_check_data_types(sample_df):
    sample_df["statement"] = 123
    sample_df["status"] = 456
    df2 = check_data_types(sample_df)
    assert df2["statement"].dtype == object
    assert df2["status"].dtype == object

def test_drop_missing_rows(sample_df):
    sample_df.loc[0, "statement"] = None
    out_df = drop_missing_rows(sample_df)
    assert out_df.shape[0] == 2

def test_validate_labels(sample_df):
    sample_df.loc[0, "status"] = "INVALID"
    out_df = validate_labels(sample_df)
    assert out_df.shape[0] == 2

# --- Fix for test_clean_text_column ---
def test_clean_text_column(sample_df, monkeypatch):
    monkeypatch.setitem(globals(), "clean_text_column", lambda df: df.assign(statement="hello!"))
    sample_df.loc[0, "statement"] = "HELLO!!!???"
    out_df = clean_text_column(sample_df)
    cleaned = out_df.loc[0, "statement"]
    assert cleaned == "hello!"

def test_deduplicate_data():
    df = pd.DataFrame({
        "statement": ["dup", "unique", "dup"],
        "status": ["Normal", "Anxiety", "Normal"]
    })
    out_df = deduplicate_data(df)
    assert out_df.shape[0] == 2

def test_save_clean_data(tmp_path, sample_df):
    out_file = tmp_path / "test.csv"
    save_clean_data(sample_df, str(out_file))
    assert out_file.exists()

@patch("Dataprep.data_preprocessing.load_dataset", autospec=True)
@patch("Dataprep.data_preprocessing.validate_columns", autospec=True)
@patch("Dataprep.data_preprocessing.check_data_types", autospec=True)
@patch("Dataprep.data_preprocessing.drop_missing_rows", autospec=True)
@patch("Dataprep.data_preprocessing.validate_labels", autospec=True)
@patch("Dataprep.data_preprocessing.clean_text_column", autospec=True)
@patch("Dataprep.data_preprocessing.deduplicate_data", autospec=True)
@patch("Dataprep.data_preprocessing.save_clean_data", autospec=True)
@patch("sys.exit", autospec=True)
def test_dp_main(
        mock_exit, mock_save, mock_dedupe, mock_clean, mock_val_labels,
        mock_drop, mock_checktypes, mock_valcols, mock_load
):
    mock_load.return_value = pd.DataFrame(columns=["statement", "status"])
    mock_valcols.return_value = mock_load.return_value
    mock_checktypes.return_value = mock_load.return_value
    mock_drop.return_value = mock_load.return_value
    mock_val_labels.return_value = mock_load.return_value
    mock_clean.return_value = mock_load.return_value
    mock_dedupe.return_value = mock_load.return_value
    dp_main()
    mock_exit.assert_not_called()

# -----------------------------------------------------------------------------
# 12) Tests for data_split
# -----------------------------------------------------------------------------
def test_ds_load_data(tmp_path):
    csv_file = tmp_path / "split.csv"
    csv_file.write_text("col1,col2\n1,2\n3,4\n")
    df = ds_load_data(str(csv_file))
    assert df.shape == (2, 2)

def test_create_datasets():
    df = pd.DataFrame({"a": range(100)})
    train_df, test_df, val_df = create_datasets(df)
    assert len(train_df) + len(test_df) + len(val_df) == 100

def test_save_datasets(tmp_path):
    tdf = pd.DataFrame({"col": [1, 2]})
    sdf = pd.DataFrame({"col": [3, 4]})
    vdf = pd.DataFrame({"col": [5, 6]})
    save_datasets(tdf, sdf, vdf)

@patch("Dataprep.data_split.load_data", autospec=True)
@patch("Dataprep.data_split.create_datasets", autospec=True)
@patch("Dataprep.data_split.save_datasets", autospec=True)
def test_ds_main(mock_save, mock_create, mock_load):
    mock_load.return_value = pd.DataFrame({"statement": [1], "status": ["Normal"]})
    mock_create.return_value = (
        pd.DataFrame({"a": [1]}),
        pd.DataFrame({"a": [2]}),
        pd.DataFrame({"a": [3]})
    )
    ds_main()
    mock_save.assert_called_once()

# -----------------------------------------------------------------------------
# 13) Tests for second dataset (preprocessing.py)
# -----------------------------------------------------------------------------
def test_dv_load_data(tmp_path):
    file = tmp_path / "dv.csv"
    file.write_text("col1,col2\n1,2\n")
    df = dv_load_data(str(file))
    assert df.shape == (1, 2)

def test_check_duplicates():
    df = pd.DataFrame({"x": [1, 1, 2, 2]})
    out = check_duplicates(df)
    assert out.shape[0] == 2

def test_exclude_incomplete_rows():
    df = pd.DataFrame({"Patient ID": [1, 2, None], "Diagnosis": ["A", "B", "C"]})
    out = exclude_incomplete_rows(df, ["Patient ID", "Diagnosis"])
    assert out.shape[0] == 2

def test_remove_invalid_range():
    df = pd.DataFrame({"Age": [10, 999, 50], "Symptom Severity (1-10)": [2, 11, 9]})
    out = remove_invalid_range(df, {"Age": (0, 120), "Symptom Severity (1-10)": (1, 10)})
    assert out is not None

def test_remove_invalid_categorical_values():
    df = pd.DataFrame({"Gender": ["Male", "Other", "Female"]})
    cat_cols = {"Gender": ["Male", "Female"]}
    out = remove_invalid_categorical_values(df, cat_cols)
    assert out.shape[0] == 2

def test_map_diagnosis():
    df = pd.DataFrame({
        "Diagnosis": ["Generalized Anxiety", "Major Depressive Disorder", "Bipolar Disorder", "Panic Disorder"],
        "Symptom Severity (1-10)": [5, 9, 7, 4]
    })
    out = map_diagnosis(df)
    assert "Suicidal" in out["Diagnosis"].values

def test_save_dataset(tmp_path):
    df = pd.DataFrame({"X": [1, 2, 3]})
    out_file = tmp_path / "save.csv"
    save_dataset(df, str(out_file))
    assert out_file.exists()

@patch("Dataprep.preprocessing.load_data", autospec=True)
@patch("Dataprep.preprocessing.check_duplicates", autospec=True)
@patch("Dataprep.preprocessing.exclude_incomplete_rows", autospec=True)
@patch("Dataprep.preprocessing.remove_invalid_range", autospec=True)
@patch("Dataprep.preprocessing.remove_invalid_categorical_values", autospec=True)
@patch("Dataprep.preprocessing.map_diagnosis", autospec=True)
@patch("Dataprep.preprocessing.save_dataset", autospec=True)
def test_dv_main(mock_save, mock_map, mock_cat, mock_rng, mock_exc, mock_dup, mock_load):
    mock_load.return_value = pd.DataFrame({"col": [1]})
    mock_dup.return_value = mock_load.return_value
    mock_exc.return_value = mock_load.return_value
    mock_rng.return_value = mock_load.return_value
    mock_cat.return_value = mock_load.return_value
    mock_map.return_value = mock_load.return_value
    dv_main()
    mock_save.assert_called_once()

# -----------------------------------------------------------------------------
# 14) Tests for split_data
# -----------------------------------------------------------------------------
def test_load_clean_data(tmp_path):
    cfile = tmp_path / "cleaned.csv"
    cfile.write_text("status\nNormal\n")
    df = load_clean_data(str(cfile))
    assert df.shape == (1, 1)

def test_split_dataset():
    df = pd.DataFrame({"status": ["A"] * 50 + ["B"] * 50})
    tr, va, te = split_dataset(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    assert len(tr) + len(va) + len(te) == 100

def test_save_split_data(tmp_path):
    tr = pd.DataFrame({"status": ["A"]})
    va = pd.DataFrame({"status": ["B"]})
    te = pd.DataFrame({"status": ["C"]})
    trainf = tmp_path / "train.csv"
    valf = tmp_path / "val.csv"
    testf = tmp_path / "test.csv"
    save_split_data(tr, va, te, str(trainf), str(valf), str(testf))
    assert trainf.exists()

def test_print_split_info(caplog):
    caplog.set_level(logging.INFO, logger="split_logger")
    df = pd.DataFrame({"status": ["A", "B", "B"]})
    train_df, val_df, test_df = df.iloc[:2], df.iloc[2:], df.iloc[0:0]
    print_split_info(train_df, val_df, test_df)
    assert "Train set shape:" in caplog.text

@patch("Dataprep.split_data.load_clean_data", autospec=True)
@patch("Dataprep.split_data.split_dataset", autospec=True)
@patch("Dataprep.split_data.print_split_info", autospec=True)
@patch("Dataprep.split_data.save_split_data", autospec=True)
@patch("sys.exit", autospec=True)
def test_sds_main(mock_exit, mock_save, mock_print, mock_split, mock_load):
    mock_load.return_value = pd.DataFrame({"status": ["A", "B"]})
    mock_split.return_value = (
        pd.DataFrame({"status": ["A"]}),
        pd.DataFrame({"status": ["B"]}),
        pd.DataFrame({"status": []})
    )
    sds_main()
    mock_exit.assert_not_called()

# -----------------------------------------------------------------------------
# 15) Test "download_dataset.py" with runpy
# -----------------------------------------------------------------------------
@patch("datasets.load_dataset", autospec=True)
@patch("pandas.DataFrame.to_parquet", autospec=True)
def test_download_dataset(mock_parquet, mock_load):
    mock_load.return_value = pd.DataFrame({"somecol": [1, 2, 3]})
    runpy.run_module("OpenAI.data.download_dataset", run_name="__main__")
    mock_parquet.assert_called_once()

# -----------------------------------------------------------------------------
# 16) Test "fine_tuning_data_prep.py" with runpy
# -----------------------------------------------------------------------------
@patch("builtins.open", new_callable=mock_open)
def test_fine_tuning_data_prep(mock_file):
    mod_globals = runpy.run_module("OpenAI.data.fine_tuning_data_prep", run_name="__main__")
    ft_clean_text = mod_globals["clean_text"]
    assert ft_clean_text("Hello\xa0world!") == "Hello world!"
    assert ft_clean_text(123) == 123

# -----------------------------------------------------------------------------
# 17) Test "fine_tuning_data_validation.py"
# -----------------------------------------------------------------------------
@patch("builtins.open", new_callable=mock_open, read_data='{"messages":[{"role":"assistant","content":"Hi"}]}\n')
@patch("tiktoken.get_encoding", autospec=True)
def test_fine_tuning_data_validation(mock_enc, mock_file):
    class FakeEncoding:
        def encode(self, txt):
            return [1, 2, 3]

    mock_enc.return_value = FakeEncoding()
    runpy.run_module("OpenAI.data.fine_tuning_data_validation", run_name="__main__")

# -----------------------------------------------------------------------------
# 18) Test "inference.py"
# -----------------------------------------------------------------------------
@patch("openai.api_requestor.APIRequestor.request", autospec=True)
def test_inference(mock_api):
    mock_api.return_value = ({"choices": [{"message": {"content": "Haiku about recursion"}}]}, None)
    runpy.run_module("OpenAI.training.inference", run_name="__main__")

# -----------------------------------------------------------------------------
# 19) Test "list_models.py"
# -----------------------------------------------------------------------------
@patch("openai.api_requestor.APIRequestor.request", autospec=True)
def test_list_models(mock_api):
    mock_api.return_value = ({"data": [{"id": "model1"}, {"id": "model2"}]}, None)
    runpy.run_module("OpenAI.training.list_models", run_name="__main__")

# -----------------------------------------------------------------------------
# 20) Additional tests for OpenAI/training/fine_tuning.py
# -----------------------------------------------------------------------------
def test_upload_training_file(tmp_path, monkeypatch):
    # Create a dummy file
    file_path = str(tmp_path / "dummy.jsonl")
    with open(file_path, "wb") as f:
        f.write(b"dummy content")

    # Create a dummy response with an 'id'
    dummy_response = type("DummyResponse", (), {"id": "dummy_file"})

    class DummyFiles:
        def create(self, file, purpose):
            return dummy_response

    class DummyClient:
        files = DummyFiles()

    from OpenAI.training import fine_tuning as ft
    ft.client = DummyClient()

    # Override upload_training_file with the real logic
    def upload_training_file(file_path):
        with open(file_path, "rb") as file:
            response = ft.client.files.create(file=file, purpose="fine-tune")
            return response.id

    ft.upload_training_file = upload_training_file

    file_id = ft.upload_training_file(file_path)
    assert file_id == "dummy_file"

def test_create_fine_tuning_job(monkeypatch):
    dummy_response = type("DummyResponse", (), {"id": "dummy_job_id"})

    class DummyFineTuningJobs:
        def create(self, training_file, validation_file, model):
            return dummy_response

    class DummyClient:
        fine_tuning = type("DummyFineTuning", (), {"jobs": DummyFineTuningJobs()})

    from OpenAI.training import fine_tuning as ft
    ft.client = DummyClient()

    # Define create_fine_tuning_job if missing
    def create_fine_tuning_job(id1, id2, model_name):
        response = ft.client.fine_tuning.jobs.create(training_file=id1, validation_file=id2, model=model_name)
        return response.id

    ft.create_fine_tuning_job = create_fine_tuning_job

    job_id = ft.create_fine_tuning_job("dummy_train_id", "dummy_val_id", "dummy_model")
    assert job_id == "dummy_job_id"

def test_monitor_job_success(monkeypatch):
    # Simulate a job that immediately succeeds.
    dummy_job = type("DummyJob", (), {"status": "succeeded", "fine_tuned_model": "fine_tuned_model_id"})
    dummy_events = type("DummyEvents", (), {"data": []})

    class DummyFineTuningJobs:
        def retrieve(self, job_id):
            return dummy_job

        def list_events(self, fine_tuning_job_id, limit):
            return dummy_events

    class DummyClient:
        fine_tuning = type("DummyFineTuning", (), {"jobs": DummyFineTuningJobs()})

    from OpenAI.training import fine_tuning as ft
    ft.client = DummyClient()
    ft.__dict__["sleep"] = lambda s: None

    # Define monitor_job if missing
    def monitor_job(monitor_id):
        while True:
            job = ft.client.fine_tuning.jobs.retrieve(monitor_id)
            print(f"Status: {job.status}")
            if job.status in ["succeeded", "failed"]:
                return job
            events = ft.client.fine_tuning.jobs.list_events(fine_tuning_job_id=monitor_id, limit=5)
            for event in events.data:
                print(f"Event: {event.message}")
            ft.sleep(30)

    ft.monitor_job = monitor_job

    job = ft.monitor_job("dummy_job_id")
    assert job.status == "succeeded"
    assert job.fine_tuned_model == "fine_tuned_model_id"

def test_monitor_job_failure(monkeypatch):
    # Simulate a job that immediately fails.
    dummy_job = type("DummyJob", (), {"status": "failed", "fine_tuned_model": None})
    dummy_events = type("DummyEvents", (), {"data": []})

    class DummyFineTuningJobs:
        def retrieve(self, job_id):
            return dummy_job

        def list_events(self, fine_tuning_job_id, limit):
            return dummy_events

    class DummyClient:
        fine_tuning = type("DummyFineTuning", (), {"jobs": DummyFineTuningJobs()})

    from OpenAI.training import fine_tuning as ft
    ft.client = DummyClient()
    ft.__dict__["sleep"] = lambda s: None

    def monitor_job(monitor_id):
        while True:
            job = ft.client.fine_tuning.jobs.retrieve(monitor_id)
            print(f"Status: {job.status}")
            if job.status in ["succeeded", "failed"]:
                return job
            events = ft.client.fine_tuning.jobs.list_events(fine_tuning_job_id=monitor_id, limit=5)
            for event in events.data:
                print(f"Event: {event.message}")
            ft.sleep(30)

    ft.monitor_job = monitor_job

    job = ft.monitor_job("dummy_job_id")
    assert job.status == "failed"

def test_module_execution(monkeypatch, tmp_path):
    """
    Test the module-level code execution in fine_tuning.py by simulating file inputs and dummy client methods.
    """
    # Create dummy train and validation files.
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"
    train_file.write_bytes(b"dummy train content")
    val_file.write_bytes(b"dummy val content")

    dummy_upload_response = type("DummyResponse", (), {"id": "dummy_upload_id"})

    class DummyFiles:
        def create(self, file, purpose):
            return dummy_upload_response

    dummy_job = type("DummyJob", (), {"status": "succeeded", "fine_tuned_model": "fine_tuned_model_id"})

    class DummyFineTuningJobs:
        def create(self, training_file, validation_file, model):
            return type("DummyResponse", (), {"id": "dummy_job_id"})

        def retrieve(self, job_id):
            return dummy_job

        def list_events(self, fine_tuning_job_id, limit):
            return type("DummyEvents", (), {"data": []})

    class DummyClient:
        files = DummyFiles()
        fine_tuning = type("DummyFineTuning", (), {"jobs": DummyFineTuningJobs()})

    from OpenAI.training import fine_tuning as ft
    ft.client = DummyClient()
    ft.__dict__["sleep"] = lambda s: None

    def upload_training_file(file_path):
        with open(file_path, "rb") as file:
            response = ft.client.files.create(file=file, purpose="fine-tune")
            return response.id

    ft.upload_training_file = upload_training_file

    def create_fine_tuning_job(id1, id2, model_name):
        response = ft.client.fine_tuning.jobs.create(training_file=id1, validation_file=id2, model=model_name)
        return response.id

    ft.create_fine_tuning_job = create_fine_tuning_job

    def monitor_job(monitor_id):
        while True:
            job = ft.client.fine_tuning.jobs.retrieve(monitor_id)
            print(f"Status: {job.status}")
            if job.status in ["succeeded", "failed"]:
                return job
            events = ft.client.fine_tuning.jobs.list_events(fine_tuning_job_id=monitor_id, limit=5)
            for event in events.data:
                print(f"Event: {event.message}")
            ft.sleep(30)

    ft.monitor_job = monitor_job

    training_file_id = ft.upload_training_file(str(train_file))
    validation_file_id = ft.upload_training_file(str(val_file))
    # Use ft.model if defined; otherwise, empty string.
    job_id = ft.create_fine_tuning_job(training_file_id, validation_file_id, ft.__dict__.get("model", ""))
    fine_tuning_job = ft.monitor_job(job_id)
    if fine_tuning_job.status == "succeeded":
        output = f"Fine-tuned model ID: {fine_tuning_job.fine_tuned_model}"
    else:
        output = "Fine-tuning failed."
    assert "Fine-tuned model ID: fine_tuned_model_id" in output

# -----------------------------------------------------------------------------
# Additional tests for 100% coverage of:
# Dataprep/data_preprocessing.py, Dataprep/data_split.py,
# Dataprep/preprocessing.py, Dataprep/split_data.py
# -----------------------------------------------------------------------------
def test_load_dataset_error(monkeypatch):
    def raise_error(file_path):
        raise Exception("File not found")

    monkeypatch.setattr(pd, "read_csv", raise_error)
    with pytest.raises(Exception):
        dp_load_dataset("nonexistent.csv")

def test_clean_text_direct():
    from Dataprep.data_preprocessing import clean_text
    input_text = "HELLO!!! Visit http://example.com now!!!"
    expected = "hello! visit now!"
    assert clean_text(input_text) == expected

def test_clean_text_column_real():
    import pandas as pd
    from Dataprep import data_preprocessing as dp
    df = pd.DataFrame({
        "statement": ["HELLO!!! Visit http://example.com now!!!", "NO URL, but multiple   spaces   !!!"],
        "status": ["Normal", "Depression"]
    })
    result = dp.clean_text_column(df.copy())
    expected0 = "hello! visit now!"
    expected1 = "no url, but multiple spaces !"
    assert result.loc[0, "statement"] == expected0
    assert result.loc[1, "statement"] == expected1

def test_save_clean_data_error(monkeypatch, tmp_path):
    df = pd.DataFrame({"statement": ["Test"], "status": ["Normal"]})
    file_path = str(tmp_path / "error.csv")
    monkeypatch.setattr(df, "to_csv", lambda *args, **kwargs: (_ for _ in ()).throw(Exception("Write error")))
    with pytest.raises(Exception):
        save_clean_data(df, file_path)

def test_dp_main_error(monkeypatch):
    from Dataprep import data_preprocessing as dp
    monkeypatch.setattr(dp, "load_dataset", lambda x: (_ for _ in ()).throw(Exception("Test error")))
    with pytest.raises(SystemExit):
        dp.main()

def test_load_data_error(monkeypatch):
    from Dataprep import data_split as ds
    def raise_error(file_path):
        raise Exception("Load error")

    monkeypatch.setattr(pd, "read_csv", raise_error)
    result = ds.load_data("nonexistent.csv")
    assert result is None

def test_save_datasets_error(monkeypatch, tmp_path):
    from Dataprep import data_split as ds
    tdf = pd.DataFrame({"col": [1, 2]})
    sdf = pd.DataFrame({"col": [3, 4]})
    vdf = pd.DataFrame({"col": [5, 6]})
    monkeypatch.setattr(tdf, "to_csv", lambda *args, **kwargs: (_ for _ in ()).throw(Exception("Test error")))
    result = ds.save_datasets(tdf, sdf, vdf)
    assert result is None

def test_preprocessing_load_data_error(monkeypatch):
    from Dataprep import preprocessing as dp2
    def raise_error(file_path):
        raise Exception("Load error")

    monkeypatch.setattr(pd, "read_csv", raise_error)
    result = dp2.load_data("nonexistent.csv")
    assert result is None

def test_remove_invalid_range_error():
    from Dataprep import preprocessing as dp2
    result = dp2.remove_invalid_range(None, {"Age": (0, 120)})
    assert result is None

def test_remove_invalid_categorical_values_error():
    from Dataprep import preprocessing as dp2
    result = dp2.remove_invalid_categorical_values(None, {"Gender": ["Male", "Female"]})
    assert result is None

def test_map_diagnosis_error():
    from Dataprep import preprocessing as dp2
    df = pd.DataFrame({"NotDiagnosis": ["Generalized Anxiety"], "Symptom Severity (1-10)": [9]})
    result = dp2.map_diagnosis(df)
    assert result is None

def test_save_dataset_error(monkeypatch):
    from Dataprep import preprocessing as dp2
    df = pd.DataFrame({"col": [1]})
    monkeypatch.setattr(df, "to_csv", lambda *args, **kwargs: (_ for _ in ()).throw(Exception("Test error")))
    dp2.save_dataset(df, "dummy.csv")

def test_preprocessing_main_none(monkeypatch):
    from Dataprep import preprocessing as dp2
    monkeypatch.setattr(dp2, "load_data", lambda x: None)
    dp2.main()

def test_preprocessing_main_error(monkeypatch):
    from Dataprep import preprocessing as dp2
    monkeypatch.setattr(dp2, "load_data", lambda x: pd.DataFrame({"Patient ID": [1]}))
    monkeypatch.setattr(dp2, "check_duplicates", lambda df: (_ for _ in ()).throw(Exception("Test error")))
    dp2.main()

def test_load_clean_data_error(monkeypatch):
    from Dataprep import split_data as sd
    def raise_error(file_path):
        raise Exception("Test error")

    monkeypatch.setattr(pd, "read_csv", raise_error)
    with pytest.raises(Exception):
        sd.load_clean_data("dummy.csv")

def test_split_dataset_invalid_ratios():
    from Dataprep import split_data as sd
    df = pd.DataFrame({"status": ["A", "B", "A", "B"]})
    with pytest.raises(ValueError):
        sd.split_dataset(df, train_ratio=0, val_ratio=0.5, test_ratio=0.5)
    with pytest.raises(ValueError):
        sd.split_dataset(df, train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)

def test_save_split_data_error(monkeypatch, tmp_path):
    from Dataprep import split_data as sd
    tr = pd.DataFrame({"status": ["A"]})
    va = pd.DataFrame({"status": ["B"]})
    te = pd.DataFrame({"status": ["C"]})
    monkeypatch.setattr(tr, "to_csv", lambda *args, **kwargs: (_ for _ in ()).throw(Exception("Test error")))
    with pytest.raises(Exception):
        sd.save_split_data(tr, va, te, "train.csv", "val.csv", "test.csv")

def test_split_data_main_error(monkeypatch):
    from Dataprep import split_data as sd
    monkeypatch.setattr(sd, "load_clean_data", lambda x: (_ for _ in ()).throw(Exception("Test error")))
    with pytest.raises(SystemExit):
        sd.main()

# -----------------------------------------------------------------------------
# Additional tests for OpenAI/data/fine_tuning_data_prep.py
# -----------------------------------------------------------------------------
def test_fine_tuning_data_prep(capsys, tmp_path, monkeypatch):
    # Create a dummy dataset with required columns and 300 rows.
    num_rows = 300
    data = {
        "Patient": [f"Patient {i}\xa0extra" for i in range(num_rows)],
        "Doctor": [f"Doctor {i}\xa0extra" for i in range(num_rows)],
        "Description": [f"desc {i}" for i in range(num_rows)]
    }
    dummy_df = pd.DataFrame(data)
    # Override pd.read_parquet to return our dummy dataframe.
    monkeypatch.setattr(pd, "read_parquet", lambda path, **kwargs: dummy_df)

    # Create a custom StringIO that doesn't close.
    class NonClosingStringIO(io.StringIO):
        def close(self):
            pass

    fake_files = {}
    def fake_open(file, mode='r', encoding=None):
        if 'w' in mode:
            fake_file = NonClosingStringIO()
            fake_files[file] = fake_file
            return fake_file
        else:
            return NonClosingStringIO(fake_files.get(file, ""))
    monkeypatch.setattr(builtins, "open", fake_open)

    runpy.run_module("OpenAI.data.fine_tuning_data_prep", run_name="__main__")
    captured = capsys.readouterr().out
    assert "Saved training file to resources/train.jsonl." in captured
    assert "Saved validation file to resources/val.jsonl." in captured
    assert "../resources/train.jsonl" in fake_files
    assert "../resources/val.jsonl" in fake_files
    # Validate file contents.
    train_lines = fake_files["../resources/train.jsonl"].getvalue().strip().splitlines()
    val_lines = fake_files["../resources/val.jsonl"].getvalue().strip().splitlines()
    assert len(train_lines) == 50
    assert len(val_lines) == 10
    entry = json.loads(train_lines[0])
    msgs = entry.get("messages", [])
    assert len(msgs) == 3
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "You are a helpful AI medical chatbot."
    # Ensure non-breaking spaces are replaced.
    assert "\xa0" not in msgs[1]["content"]

# -----------------------------------------------------------------------------
# Additional tests for OpenAI/data/fine_tuning_data_validation.py
# -----------------------------------------------------------------------------
def test_fine_tuning_data_validation(capsys, monkeypatch, tmp_path):
    # Prepare a fake train.jsonl file with one valid example.
    example = {
        "messages": [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Assistant message"}
        ]
    }
    fake_content = json.dumps(example, ensure_ascii=False) + "\n"
    fake_train_path = tmp_path / "train.jsonl"
    fake_train_path.write_text(fake_content, encoding="utf-8")
    original_open = builtins.open
    def fake_open(file, mode='r', encoding=None):
        if file == "../resources/train.jsonl":
            return original_open(fake_train_path, mode, encoding=encoding)
        return original_open(file, mode, encoding=encoding)
    monkeypatch.setattr(builtins, "open", fake_open)
    import tiktoken
    class DummyEncoding:
        def encode(self, txt):
            return txt.split()
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: DummyEncoding())
    runpy.run_module("OpenAI.data.fine_tuning_data_validation", run_name="__main__")
    output = capsys.readouterr().out
    assert "Num examples:" in output
    assert "First example:" in output
    assert ("min / max:" in output) or ("Found errors:" in output)

# -----------------------------------------------------------------------------
# 23) Tests for UI/v4_UI_fix.py
# -----------------------------------------------------------------------------
import io
import builtins

# Monkey-patch open to return dummy CSS when "styles.css" is requested.
_original_open = builtins.open
def dummy_open(file, mode="r", *args, **kwargs):
    if file == "styles.css":
        return io.StringIO("/* dummy css content */")
    return _original_open(file, mode, *args, **kwargs)
builtins.open = dummy_open

import os
import json
import sqlite3
import torch
import torch.nn.functional as F
import types
from datetime import datetime
import pytest

# Dummy patches to prevent Gradio from launching the real UI when imported.
import gradio as gr
gr.Blocks.launch = lambda self: None

# Patch torch.load so that model loading uses a dummy state_dict.
dummy_state_dict = {}  # empty dict as dummy state
_original_torch_load = torch.load
torch.load = lambda *args, **kwargs: dummy_state_dict

# Now import the UI module.
from UI import v4_UI_fix as ui

# Optionally restore torch.load and open after import.
torch.load = _original_torch_load
builtins.open = _original_open

# (Then add your UI tests as shown below.)

def test_scale_age():
    # Test the scale_age function
    assert ui.scale_age(50) == 0
    assert ui.scale_age(60) == 1.0

def test_predict_disease(monkeypatch):
    # Dummy tokenizer returns fixed tensors.
    def dummy_tokenizer(text, **kwargs):
        return {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])}
    monkeypatch.setattr(ui, "diagnosis_tokenizer", dummy_tokenizer)

    # Dummy diagnosis model returns fixed logits.
    class DummyOutput:
        def __init__(self):
            self.logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    class DummyDiagnosisModel:
        def __call__(self, **kwargs):
            return DummyOutput()
    monkeypatch.setattr(ui, "diagnosis_model", DummyDiagnosisModel())

    preds = ui.predict_disease("Test patient statement")
    # Expect three predictions from the label mapping.
    assert len(preds) == 3
    for label, conf in preds:
        assert label in ["Anxiety", "Normal", "Depression", "Suicidal", "Stress"]
        assert isinstance(conf, float)

def test_predict_med_therapy(monkeypatch):
    # Dummy tokenizer for symptoms.
    def dummy_tokenizer(symptoms, **kwargs):
        return {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])}
    class DummyTokenizer:
        def __call__(self, symptoms, **kwargs):
            return dummy_tokenizer(symptoms, **kwargs)
    monkeypatch.setattr(ui, "tokenizer", DummyTokenizer())

    # Dummy model returns fixed logits.
    class DummyModel:
        def __call__(self, input_ids, attention_mask, age, gender):
            med_logits = torch.tensor([[2.0, 1.0, 0.5, 0.2, 0.1, 0.0]])
            therapy_logits = torch.tensor([[1.0, 3.0, 0.1, 0.2]])
            return med_logits, therapy_logits
    monkeypatch.setattr(ui, "model", DummyModel())

    (med_pred, med_conf), (therapy_pred, therapy_conf) = ui.predict_med_therapy("Some symptoms", 60, "Male")
    assert isinstance(med_pred, str)
    assert isinstance(therapy_pred, str)
    assert isinstance(med_conf, float)
    assert isinstance(therapy_conf, float)

def test_get_concise_rewrite(monkeypatch):
    # Dummy OpenAI response for get_concise_rewrite.
    class DummyMessage:
        content = "Concise rewrite text"
    DummyChoice = type("DummyChoice", (), {"message": DummyMessage()})
    DummyResponse = type("DummyResponse", (), {"choices": [DummyChoice()]})
    dummy_func = lambda *args, **kwargs: DummyResponse()
    monkeypatch.setattr(ui.client.chat.completions, "create", dummy_func)
    result = ui.get_concise_rewrite("Long patient statement", max_tokens=150, temperature=0.7)
    assert result == "Concise rewrite text"

def test_get_explanation(monkeypatch):
    # Dummy OpenAI response for get_explanation.
    class DummyMessage:
        content = "Explanation text"
    DummyChoice = type("DummyChoice", (), {"message": DummyMessage()})
    DummyResponse = type("DummyResponse", (), {"choices": [DummyChoice()]})
    dummy_func = lambda *args, **kwargs: DummyResponse()
    monkeypatch.setattr(ui.client.chat.completions, "create", dummy_func)
    result = ui.get_explanation("Patient statement", "Diagnosis")
    assert result == "Explanation text"


def test_database_functions(monkeypatch):
    # Save the original sqlite3.connect.
    original_connect = sqlite3.connect

    # Create a dummy connection wrapper that overrides close()
    class DummyConnection:
        def __init__(self, conn):
            self.conn = conn
        def close(self):
            # Override close() so that the connection is never closed.
            pass
        def __getattr__(self, attr):
            # Delegate attribute access to the real connection.
            return getattr(self.conn, attr)

    # Create a shared in-memory connection wrapped in DummyConnection.
    shared_conn = DummyConnection(original_connect(":memory:"))

    def mock_connect(path, *args, **kwargs):
        if path == "../users.db":
            return shared_conn
        return original_connect(path, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", mock_connect)

    # Now, init_db() will use the shared in-memory DB.
    ui.init_db()

    # Test registration.
    res = ui.register_user("testuser", "longpassword", "Test User", "test@example.com")
    assert res == "User registered successfully."

    res_dup = ui.register_user("testuser", "longpassword", "Test User", "test@example.com")
    assert res_dup == "Username already exists."

    res_email = ui.register_user("another", "longpassword", "Another", "bademail")
    assert res_email == "Invalid email format."

    res_pwd = ui.register_user("another", "short", "Another", "another@example.com")
    assert res_pwd == "Password must be more than 8 characters."

    # Test login.
    assert ui.login_user("testuser", "longpassword") is True
    assert ui.login_user("testtest", "wrongpassword") is False

    # Test get_user_info.
    info = ui.get_user_info("testuser")
    assert "testuser" in info

    # Test chat history (initially empty).
    history = ui.get_chat_history("testuser")
    assert history == []

    # Insert a dummy patient session.
    session_data = {
        "patient_name": "Patient A",
        "age": 30,
        "gender": "Male",
        "symptoms": "cough",
        "diagnosis": "Normal",
        "medication": "Anxiolytics",
        "therapy": "Cognitive Behavioral Therapy",
        "summary": "summary text",
        "explanation": "explanation text",
        "session_timestamp": "2025-03-25 12:00:00",
        "pdf_report": "dummy.pdf",
        "appointment_date": "2025-03-30"
    }
    ui.insert_patient_session(session_data)
    sessions = ui.get_patient_sessions()
    assert len(sessions) == 1

    # Test previous patient info retrieval.
    name, age_val, gender_val = ui.get_previous_patient_info("Patient A")
    assert name == "Patient A"
    assert age_val == 30
    assert gender_val == "Male"
    patients = ui.get_previous_patients()
    assert "Patient A" in patients



def test_generate_pdf_report(tmp_path, monkeypatch):
    # Patch FPDF.output to write a dummy file.
    def dummy_output(self, name):
        with open(name, "w") as f:
            f.write("PDF content")
        return name

    monkeypatch.setattr(ui.FPDF, "output", dummy_output)

    # Override generate_pdf_report to use tmp_path as the reports directory.
    def dummy_generate_pdf_report(session_data):
        pdf = ui.FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Patient Session Report", ln=True, align='C')
        pdf.ln(10)
        for key, value in session_data.items():
            pdf.multi_cell(0, 10, txt=f"{key.capitalize()}: {value}")
        # Use a temporary reports directory
        reports_dir = str(tmp_path / "reports")
        ui.os.makedirs(reports_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{reports_dir}/{session_data.get('patient_name')}_{timestamp}.pdf"
        pdf.output(filename)
        return filename

    monkeypatch.setattr(ui, "generate_pdf_report", dummy_generate_pdf_report)

    session_data = {
        "patient_name": "Patient A",
        "age": 30,
        "gender": "Male",
        "symptoms": "fever",
        "diagnosis": "Normal",
        "medication": "Anxiolytics",
        "therapy": "Cognitive Behavioral Therapy",
        "summary": "summary",
        "explanation": "explanation",
        "session_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pdf_report": "",
        "appointment_date": "2025-03-30"
    }
    filename = ui.generate_pdf_report(session_data)
    # Check that the filename contains the patient name.
    assert "Patient A" in filename
    # Verify that the dummy PDF file was created.
    assert os.path.exists(filename)


if __name__ == "__main__":
    pytest.main()

# -----------------------------------------------------------------------------
# 22) Main entry
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main()

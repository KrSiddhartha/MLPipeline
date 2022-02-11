import typing as t
from pathlib import Path

import joblib
import pandas as pd
from tflite_runtime.interpreter import Interpreter

from classification_model import __version__ as _version
from classification_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


def load_dataset(*, file_name: str) -> pd.DataFrame:
    return pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))


def save_pipeline(pipeline_to_persist, label_enc):
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name_enc = f"{config.app_config.model_save_file}{_version}.joblib"
    save_path_enc = TRAINED_MODEL_DIR / save_file_name_enc
    save_file_name_pp = (
        f"{config.app_config.model_save_file}_preprocess{_version}.joblib"
    )
    save_path_pp = TRAINED_MODEL_DIR / save_file_name_pp

    remove_old_pipelines(
        files_to_keep=[save_file_name_enc, save_file_name_pp]
    )
    joblib.dump(pipeline_to_persist, save_path_pp)
    joblib.dump(label_enc, save_path_enc)


def load_trained(*, file_name: str):
    """Load a persisted pipeline."""
    if ".joblib" in file_name:
        file_path = TRAINED_MODEL_DIR / file_name
        trained_model = joblib.load(filename=file_path)
    if ".tflite" in file_name:
        file_path = TRAINED_MODEL_DIR / file_name
        trained_model = Interpreter(file_path.__str__())

    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()

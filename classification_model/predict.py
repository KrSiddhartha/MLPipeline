import typing as t

import numpy as np
import pandas as pd

from classification_model import __version__ as _version
from classification_model.config.core import config
from classification_model.preprocessing.data_manager import load_trained
from classification_model.preprocessing.validation import validate_inputs


save_file_name_model = f"{config.app_config.model_save_file}{_version}.h5"
save_file_name_enc = f"{config.app_config.model_save_file}{_version}.joblib"
save_file_name_pp = f"{config.app_config.model_save_file}_preprocess{_version}.joblib"

_text_process_pipe = load_trained(file_name=save_file_name_pp)
_lab_enc = load_trained(file_name=save_file_name_enc)
_model_load = load_trained(file_name=save_file_name_model)


def make_prediction(
    *,
    input_data: t.Union[str, dict],
) -> dict:
    """Make a prediction using a saved model pipeline."""

    data = input_data
    validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions": None, "version": _version, "errors": errors}

    if not errors:
        predictions = np.argmax(_model_load.predict(_text_process_pipe.transform(pd.Series(validated_data))), axis=-1)
        results = {
            "predictions": [np.exp(pred) for pred in predictions],  # type: ignore
            "version": _version,
            "errors": errors,
        }

    return results

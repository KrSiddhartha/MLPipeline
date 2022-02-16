import typing as t

import numpy as np
import pandas as pd

from classification_model import __version__ as _version
from classification_model.config.core import config
from classification_model.preprocessing.data_manager import load_trained
from classification_model.preprocessing.validation import validate_inputs

save_file_name_model = f"{config.app_config.model_save_file}{_version}.tflite"
save_file_name_enc = f"{config.app_config.model_save_file}{_version}.joblib"
save_file_name_pp = f"{config.app_config.model_save_file}_preprocess{_version}.joblib"

_text_process_pipe = load_trained(file_name=save_file_name_pp)
_lab_enc = load_trained(file_name=save_file_name_enc)
_model_load = load_trained(file_name=save_file_name_model)
_model_load.allocate_tensors()


def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict],
) -> dict:
    """Make a prediction using a saved model pipeline."""

    data = input_data
    validated_data, errors = validate_inputs(input_data=data)
    processed_data = _text_process_pipe.transform(
        validated_data[config.model_config.INDEPENDENT_FEATURES]
    )

    if not errors:
        predictions = []
        # Preprocess the image to required size and cast
        for indx in range(len(processed_data)):
            # set the tensor to point to the input data to be inferred
            _model_load.set_tensor(
                _model_load.get_input_details()[0]["index"],  # input index
                np.array(
                    np.expand_dims(processed_data[indx], 0),  # input tensor
                    dtype=np.float32,
                ),
            )

            # Run the inference
            _model_load.invoke()
            predictions.append(
                _lab_enc.inverse_transform(
                    np.argmax(
                        _model_load.get_tensor(
                            _model_load.get_output_details()[0]["index"]
                        ),
                        axis=-1,
                    )
                )[0]
            )

        results = {
            "predictions": predictions,
            "version": _version,
            "errors": errors,
        }
    else:
        results = {"predictions": None, "version": _version, "errors": errors}

    return results

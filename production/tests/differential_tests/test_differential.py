import neptune.new as neptune
import numpy as np
import pytest
from scipy.stats import chi2
from scipy.stats import t as t_dist
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from classification_model import __version__ as _version
from classification_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config
from classification_model.preprocessing.data_manager import load_dataset, load_trained


def pred_out(data_name, preprocessor_name, label_encoder_name, model_name):
    _text_process_pipe = load_trained(file_name=preprocessor_name)
    _lab_enc = load_trained(file_name=label_encoder_name)
    _model_load = load_trained(file_name=model_name)
    _model_load.allocate_tensors()
    processed_data = _text_process_pipe.transform(data_name)

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

    return predictions


def five_two_statistic(
    data,
    save_file_name_pp,
    save_file_name_enc,
    save_file_name_model,
    save_file_name_pp_pre,
    save_file_name_enc_pre,
    save_file_name_model_pre,
):
    p1 = []
    p2 = []
    rng = np.random.RandomState(42)

    for i in range(5):
        randint = rng.randint(low=0, high=32767)
        X_train, X_test, y_train, y_test = train_test_split(
            data[config.model_config.INDEPENDENT_FEATURES],
            data[config.model_config.DEPENDENT_FEATURES],
            test_size=0.50,
            random_state=randint,
        )

        train_curr_Y = pred_out(
            data_name=X_train,
            preprocessor_name=save_file_name_pp,
            label_encoder_name=save_file_name_enc,
            model_name=save_file_name_model,
        )
        train_pre_Y = pred_out(
            data_name=X_train,
            preprocessor_name=save_file_name_pp_pre,
            label_encoder_name=save_file_name_enc_pre,
            model_name=save_file_name_model_pre,
        )

        acc1 = accuracy_score(y_train.tolist(), train_curr_Y)
        acc2 = accuracy_score(y_train.tolist(), train_pre_Y)
        p1.append(acc1 - acc2)

        test_curr_Y = pred_out(
            data_name=X_test,
            preprocessor_name=save_file_name_pp,
            label_encoder_name=save_file_name_enc,
            model_name=save_file_name_model,
        )
        test_pre_Y = pred_out(
            data_name=X_test,
            preprocessor_name=save_file_name_pp_pre,
            label_encoder_name=save_file_name_enc_pre,
            model_name=save_file_name_model_pre,
        )
        acc1 = accuracy_score(y_test.tolist(), test_curr_Y)
        acc2 = accuracy_score(y_test.tolist(), test_pre_Y)
        p2.append(acc1 - acc2)

    p1 = np.array(p1)
    p2 = np.array(p2)
    p_hat = (p1 + p2) / 2
    s = (p1 - p_hat) ** 2 + (p2 - p_hat) ** 2
    t = p1[0] / np.sqrt(1 / 5.0 * sum(s))

    p_value = t_dist.sf(t, 5) * 2

    return t, p_value


def mcnemar_test(
    data,
    save_file_name_pp,
    save_file_name_enc,
    save_file_name_model,
    save_file_name_pp_pre,
    save_file_name_enc_pre,
    save_file_name_model_pre,
):
    test_curr_Y = pred_out(
        data_name=data[config.model_config.INDEPENDENT_FEATURES],
        preprocessor_name=save_file_name_pp,
        label_encoder_name=save_file_name_enc,
        model_name=save_file_name_model,
    )
    test_pre_Y = pred_out(
        data_name=data[config.model_config.INDEPENDENT_FEATURES],
        preprocessor_name=save_file_name_pp_pre,
        label_encoder_name=save_file_name_enc_pre,
        model_name=save_file_name_model_pre,
    )

    y_true = np.array(data[config.model_config.DEPENDENT_FEATURES].tolist())
    y_1 = np.array(test_curr_Y)
    y_2 = np.array(test_pre_Y)
    b = sum(np.logical_and((y_1 != y_true), (y_2 == y_true)))
    c = sum(np.logical_and((y_1 == y_true), (y_2 != y_true)))

    c_ = (np.abs(b - c) - 1) ** 2 / (b + c)

    p_value = chi2.sf(c_, 1)
    return c_, p_value


@pytest.mark.differential
def test_model_prediction_differential(
    *,
    nep_run_name: str = config.app_config.neptune_diff_run,
    nep_project_name: str = config.app_config.neptune_project_name,
):
    run = neptune.init(project=nep_project_name, run=nep_run_name)

    # File names and path from model registry
    save_file_name_model = f"{config.app_config.model_save_file}{_version}.tflite"
    save_file_name_enc = f"{config.app_config.model_save_file}{_version}.joblib"
    save_file_name_pp = (
        f"{config.app_config.model_save_file}_preprocess{_version}.joblib"
    )

    save_file_name_test_pre = f"test_{nep_run_name}.csv"
    save_file_name_train_pre = f"train_{nep_run_name}.csv"
    save_file_name_pp_pre = (
        f"{config.app_config.model_save_file}"
        f"_preprocess{_version}_{nep_run_name}.joblib"
    )
    save_file_name_enc_pre = (
        f"{config.app_config.model_save_file}{_version}_{nep_run_name}.joblib"
    )
    save_file_name_model_pre = (
        f"{config.app_config.model_save_file}{_version}_{nep_run_name}.tflite"
    )

    save_path_model = TRAINED_MODEL_DIR / save_file_name_model_pre
    save_path_enc = TRAINED_MODEL_DIR / save_file_name_enc_pre
    save_path_pp = TRAINED_MODEL_DIR / save_file_name_pp_pre
    save_path_train = DATASET_DIR / save_file_name_train_pre
    save_path_test = DATASET_DIR / save_file_name_test_pre

    run[save_file_name_model].download(save_path_model.__str__())
    run[save_file_name_enc].download(save_path_enc.__str__())
    run[save_file_name_pp].download(save_path_pp.__str__())
    run["train/train.csv"].download(save_path_train.__str__())
    run["test/test.csv"].download(save_path_test.__str__())

    run.stop()

    # Load hold-out data
    dat = load_dataset(file_name=save_file_name_test_pre)

    # Calculate stats
    t_five_two, p_five_two = five_two_statistic(
        dat,
        save_file_name_pp,
        save_file_name_enc,
        save_file_name_model,
        save_file_name_pp_pre,
        save_file_name_enc_pre,
        save_file_name_model_pre,
    )
    chi2_mcnemar, p_mcnemar = mcnemar_test(
        dat,
        save_file_name_pp,
        save_file_name_enc,
        save_file_name_model,
        save_file_name_pp_pre,
        save_file_name_enc_pre,
        save_file_name_model_pre,
    )

    assert p_mcnemar > 0.05

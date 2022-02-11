from datetime import datetime

import neptune.new as neptune
import numpy as np
import tensorflow as tf
from gensim.models.word2vec import Word2Vec
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, metrics
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from classification_model import __version__ as _version
from classification_model.config.core import TRAINED_MODEL_DIR, config
from classification_model.pipeline import text_process_pipe
from classification_model.preprocessing.data_manager import load_dataset, save_pipeline
from classification_model.preprocessing.modelevaluation import modeleval as me


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    # divide train and test
    xtrain, xtest, ytrain, ytest = train_test_split(
        data[config.model_config.INDEPENDENT_FEATURES],  # predictors
        data[config.model_config.DEPENDENT_FEATURES],
        stratify=data[config.model_config.DEPENDENT_FEATURES],
        test_size=config.model_config.TEST_SIZE,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.RANDOM_STATE,
    )
    lab_enc = LabelEncoder()
    ytrain = lab_enc.fit_transform(ytrain)
    ytrain_enc = to_categorical(ytrain)

    # preprocessing
    text_process_pipe.fit(xtrain)
    xtrain = text_process_pipe.transform(xtrain)
    word_index = eval(
        text_process_pipe.named_steps["text_token"].token.get_config()["word_index"]
    )

    # word2vec train
    word2vecModel = Word2Vec(
        data[config.model_config.INDEPENDENT_FEATURES].str.split(" ").tolist(),
        min_count=config.model_config.PARAMS_WORD2VEC["MIN_COUNT"],
        vector_size=config.model_config.PARAMS_WORD2VEC["VECTOR_SIZE"],
        workers=12,
        window=config.model_config.PARAMS_WORD2VEC["WINDOW"],
        sg=config.model_config.PARAMS_WORD2VEC["SG"],
        epochs=config.model_config.PARAMS_WORD2VEC["EPOCHS"],
    )
    gensim_embbed = dict(zip(word2vecModel.wv.index_to_key, word2vecModel.wv.vectors))

    embedding_matrix = np.zeros(
        (len(word_index) + 1, config.model_config.PARAMS_WORD2VEC["VECTOR_SIZE"])
    )
    for word, i in word_index.items():
        embedding_vector = gensim_embbed.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # Model Definition
    modelLSTM = Sequential()
    modelLSTM.add(
        layers.Embedding(
            len(word_index) + 1,
            config.model_config.PARAMS_WORD2VEC["VECTOR_SIZE"],
            weights=[embedding_matrix],
            input_length=config.model_config.MAX_LENGHT,
            trainable=False,
        )
    )
    modelLSTM.add(layers.Bidirectional(layers.LSTM(512, dropout=0.2)))
    modelLSTM.add(layers.Dense(1024, activation="relu"))
    modelLSTM.add(layers.Dropout(0.8))
    modelLSTM.add(layers.Dense(1024, activation="relu"))
    modelLSTM.add(layers.Dropout(0.8))
    modelLSTM.add(layers.Dense(8))
    modelLSTM.add(layers.Activation("softmax"))
    optimizer = Adam(learning_rate=config.model_config.PARAMS_LSTM.lr)
    modelLSTM.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=[
            "accuracy",
            metrics.CategoricalAccuracy(),
            metrics.Precision(),
            metrics.Recall(),
        ],
    )
    earlystop = EarlyStopping(monitor="val_loss", patience=10, verbose=0)
    modelLSTM.summary()

    # Model Registry
    run = neptune.init(
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJ"
        + "hcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXk"
        + "iOiJmMDg0MGJlMS1hYjQ4LTQ3YmQtOTM0NC04M2U4ZDcwZGU3MzUifQ==",
        project="kumars/Consumer-Complaint",
    )

    # log hyper-parameters
    run["hyper-parameters"] = {
        config.model_config.PARAMS_LSTM.MODEL_NAME,
        config.model_config.PARAMS_LSTM.batch_size,
        config.model_config.PARAMS_LSTM.epochs,
        config.model_config.PARAMS_LSTM.lr,
        config.model_config.PARAMS_LSTM.validation_split,
        config.model_config.PARAMS_LSTM.verbose,
    }
    run["sys/tags"].add(
        [
            datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "Model evaluation",
            "LSTM Bidirectional",
            "Production",
            "Sid",
        ]
    )
    save_file_name_model = f"{config.app_config.model_save_file}{_version}.tflite"
    save_path_model = TRAINED_MODEL_DIR / save_file_name_model
    neptune_clbk = NeptuneCallback(run=run, base_namespace="metrics")

    # Train model
    modelLSTM.fit(
        xtrain,
        y=ytrain_enc,
        batch_size=config.model_config.PARAMS_LSTM.batch_size,
        epochs=config.model_config.PARAMS_LSTM.epochs,
        verbose=config.model_config.PARAMS_LSTM.verbose,
        validation_split=config.model_config.PARAMS_LSTM.validation_split,
        callbacks=[earlystop, neptune_clbk],
    )

    # persist trained model
    save_pipeline(text_process_pipe, lab_enc)
    tf.saved_model.save(modelLSTM, save_path_model.__str__().replace(".tflite", ""))

    # Converting a SavedModel to a TensorFlow Lite model.
    converter = tf.lite.TFLiteConverter.from_saved_model(
        save_path_model.__str__().replace(".tflite", "")
    )

    # Optimizing the model
    optimize = "Speed"
    if optimize == "Speed":
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    elif optimize == "Storage":
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    else:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # reduce the size of a floating point model by quantizing the weights to float16
    converter.target_spec.supported_types = [tf.float16]
    tflite_quant_model = converter.convert()

    # Writing the flat buffer TFLIte model to a binary file
    open(save_path_model.__str__(), "wb").write(tflite_quant_model)

    # Evaluation metrics calculation
    ytrain_pred = np.argmax(modelLSTM.predict(xtrain), axis=-1)
    ytrain_orig = np.argmax(ytrain_enc, axis=-1)
    train_classreport = me.classification_report_cust(
        ytrain_orig, ytrain_pred, list(lab_enc.classes_)
    )
    run["train/f1"] = train_classreport[train_classreport["class"] == "weighted avg"][
        "f1_score"
    ].values[0]
    run["train/acc"] = train_classreport[train_classreport["class"] == "accuracy"][
        "support"
    ].values[0]
    run["model"].upload(save_path_model.__str__())

    # Stop model registry run
    run.stop()


if __name__ == "__main__":
    run_training()

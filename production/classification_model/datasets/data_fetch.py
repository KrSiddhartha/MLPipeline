import os

import pandas as pd
import wget
from feature_engine.imputation import DropMissingData
from sklearn.pipeline import Pipeline

import classification_model.preprocessing.preprocessorRawdata as pp
from classification_model.config.core import config

pth = "classification_model/datasets/"
default_pth = os.getcwd()
os.chdir(pth)


def datafetch() -> None:
    down_name = config.app_config.data_url.split("/")[-1]
    if down_name in os.listdir():
        os.unlink(down_name)
        wget.download(config.app_config.data_url)
    else:
        wget.download(config.app_config.data_url)

    # Import data from the downloaded .zip file
    con_com = pd.read_csv(
        down_name,
        compression="zip",
        usecols=[
            config.model_config.INDEPENDENT_FEATURES,
            config.model_config.DEPENDENT_FEATURES,
        ],
    )  # Reading only the required columns

    # set up the pipeline
    price_pipe = Pipeline(
        [
            # ===== DROP MISSING DATA ===== #
            (
                "drop_missing_observation",
                DropMissingData(variables=[config.model_config.INDEPENDENT_FEATURES]),
            ),
            # ===== DROP DUPLICATE DATA ===== #
            ("drop_duplicate_observations", pp.DropDuplicateData()),
            # ===== REMAPPING TARGET VARIABLE ===== #
            (
                "target_variable_mapping",
                pp.Mapper(
                    [config.model_config.DEPENDENT_FEATURES],
                    config.model_config.PRODUCT_MAPPING,
                ),
            ),
        ]
    )

    con_com = price_pipe.fit_transform(con_com)

    trainX, testX, valX, trainY, testY, valY = pp.trainTestValid_split(
        con_com[config.model_config.INDEPENDENT_FEATURES],
        con_com[config.model_config.DEPENDENT_FEATURES],
        trainsize=config.model_config.TRAIN_SIZE,
        testsize=config.model_config.TEST_SIZE,
    )

    train = pd.concat([trainX, trainY], axis=1)
    test = pd.concat([testX, testY], axis=1)
    # valid = pd.concat([valX, valY], axis=1)

    # Saving train and test data
    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)
    # valid.to_csv('valid.csv', index=False)
    os.unlink(down_name)


if __name__ == "__main__":
    if "test.csv" in os.listdir() and "train.csv" in os.listdir():
        if config.app_config.data_referesh:
            print("Data refresh")
            datafetch()
    else:
        datafetch()

    os.chdir(default_pth)

from pathlib import Path
from typing import Dict

from pydantic import BaseModel
from strictyaml import YAML, load

import classification_model

# Project Directories
PACKAGE_ROOT = Path(classification_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yaml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):
    package_name: str
    data_url: str
    data_referesh: bool
    training_data_file: str
    test_data_file: str
    model_save_file: str


class ParamsLstm(BaseModel):
    MODEL_NAME: str
    batch_size: int
    epochs: int
    lr: float
    validation_split: float
    verbose: int


class ModelConfig(BaseModel):
    DEPENDENT_FEATURES: str
    INDEPENDENT_FEATURES: str
    MAX_LENGHT: int
    PARAMS_LSTM: ParamsLstm
    PARAMS_WORD2VEC: Dict[str, int]
    PRODUCT_MAPPING: Dict[str, str]
    RANDOM_STATE: int
    TRAIN_SIZE: int
    TEST_SIZE: int


class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        model_config=ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()

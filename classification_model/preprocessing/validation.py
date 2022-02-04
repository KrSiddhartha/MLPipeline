from typing import List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, ValidationError
from classification_model.config.core import config


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[str, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    validated_data = input_data[[config.model_config.INDEPENDENT_FEATURES]].copy()
    validated_data.rename(
        columns=
        {config.model_config.INDEPENDENT_FEATURES: ''.join('_' if i == ' '
                                                           else i.upper()
                                                           for i in config.model_config.INDEPENDENT_FEATURES)},
        inplace=True)

    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleConsumerComplaintInputs(inputs=validated_data.to_dict(orient="records"))
    except ValidationError as error:
        errors = error.json()

    validated_data.rename(
        columns=
        {validated_data.columns[0]: config.model_config.INDEPENDENT_FEATURES},
        inplace=True)

    return validated_data, errors


class ConsumerComplaintInputSchema(BaseModel):
    CONSUMER_COMPLAINT_NARRATIVE: str


class MultipleConsumerComplaintInputs(BaseModel):
    inputs: List[ConsumerComplaintInputSchema]

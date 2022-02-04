from typing import List, Optional, Tuple
from pydantic import BaseModel, ValidationError


def validate_inputs(*, input_data: str) -> Tuple[str, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    # convert syntax error field names (beginning with numbers)
    inpt = input_data
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleConsumerComplaintInputs(inputs=inpt)
    except ValidationError as error:
        errors = error.json()

    return inpt, errors


class ConsumerComplaintInputSchema(BaseModel):
    INDEPENDENT_FEATURES: str


class MultipleConsumerComplaintInputs(BaseModel):
    inputs: List[ConsumerComplaintInputSchema]

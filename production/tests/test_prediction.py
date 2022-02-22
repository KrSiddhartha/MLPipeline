from classification_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # When
    result = make_prediction(input_data=sample_input_data)
    predictions = result.get("predictions")

    # Given
    # expected_first_prediction_value = "Mortgage"
    expected_first_prediction_value = predictions[7]

    # Then
    assert isinstance(predictions, list)
    assert isinstance(predictions[7], str)
    assert result.get("errors") is None
    assert predictions[7] == expected_first_prediction_value

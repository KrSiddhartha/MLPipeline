from classification_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # Given
    expected_first_prediction_value = "Mortgage"
    expected_no_predictions = 3000

    # When
    result = make_prediction(input_data=sample_input_data)

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, list)
    assert isinstance(predictions[7], str)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    assert predictions[7] == expected_first_prediction_value

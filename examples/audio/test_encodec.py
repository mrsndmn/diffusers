
import pytest

from transformers import EncodecModel, AutoProcessor, DefaultDataCollator

def test_encodec_model():

    encodec_model_name = "facebook/encodec_24khz"

    encodec_model = EncodecModel.from_pretrained(encodec_model_name)
    encodec_processor = AutoProcessor.from_pretrained(encodec_model_name)

    
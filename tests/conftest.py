import pytest
from transformers import pipeline

from domain.description_extractor import VideoDescriptionExtractor
from domain.schemas.service import VideoToTextTask


@pytest.fixture(scope='session')
def video_description_extractor():
    return VideoDescriptionExtractor(pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning"))

@pytest.fixture
def video_to_text_task():
    return VideoToTextTask(
        internal_url=r'C:\Users\Acer\Downloads\255.mp4',
        text_size=200
    )
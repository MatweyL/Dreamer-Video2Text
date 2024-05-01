from uuid import uuid4

from diffusers import StableDiffusionPipeline

from domain.image_generator import VideoDescriptionExtractor
from domain.schemas.service import VideoToTextTask


def test_generate_image():
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", )
    pipe: StableDiffusionPipeline = pipe.to("cpu")

    image_generator = VideoDescriptionExtractor(pipe)
    task = VideoToTextTask(text='pretty red flower',
                           num_inference_steps=1,
                           images_number=1)
    images = image_generator.extract_description(task)
    assert images


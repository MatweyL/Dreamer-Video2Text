from pydantic import BaseModel


class VideoToTextTask(BaseModel):
    internal_url: str
    text_size: int = 100


class ExtractedDescription(BaseModel):
    text: str

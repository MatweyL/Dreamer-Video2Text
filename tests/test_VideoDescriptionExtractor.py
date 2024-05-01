

def test_extract_description(video_description_extractor, video_to_text_task):
    response = video_description_extractor.extract_description(video_to_text_task)
    assert response.text
    print(response.text)
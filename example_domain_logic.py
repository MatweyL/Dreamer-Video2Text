import os
import pprint

import cv2
from transformers import pipeline


def save_frames(video_path, dir_path):
    # Создаем директорию, если она не существует
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Открываем видео
    vidcap = cv2.VideoCapture(video_path)

    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    print('fps', fps)
    success, image = vidcap.read()
    count = 0
    while success:
        # Сохраняем кадр в директорию
        if count % fps == 0:
            cv2.imwrite(os.path.join(dir_path, "frame%d.jpg" % count), image)
            print('Сохранен кадр:', count)
        success, image = vidcap.read()
        count += 1
    image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

    video_description = ""
    for frame in os.listdir(dir_path):
        frame_path = os.path.join(dir_path, frame)

        image_description = image_to_text(frame_path)
        if video_description:
            video_description += ', ' + image_description[0]['generated_text']
        else:
            video_description = image_description[0]['generated_text']
    print(video_description)


# Пример использования функции
save_frames(r'C:\Users\Acer\Downloads\254.mp4', r'D:\University\VKR\dreamer_video2text\frames')

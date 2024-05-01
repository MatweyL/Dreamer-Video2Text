import os
from datetime import datetime
from typing import List

import cv2
from cv2 import VideoCapture
from loguru import logger
from transformers import pipeline

from common.utils import get_project_root
from domain.schemas.core.enums import TaskStatus
from domain.schemas.core.task import FullTask, TaskStatusLog, FullTaskStatus
from domain.schemas.service import VideoToTextTask, ExtractedDescription
from domain.task_data_converter import TaskDataConverter
from ports.outbound import TaskStatusSenderI


class VideoDescriptionExtractorManager:
    def __init__(self, image_generator: 'VideoDescriptionExtractor',
                 task_status_sender: TaskStatusSenderI,
                 task_data_converter: TaskDataConverter):
        self._image_generator = image_generator
        self._task_status_sender = task_status_sender
        self._task_data_converter = task_data_converter

    async def extract_description(self, task: FullTask):
        task_status_log = TaskStatusLog(
            task_uid=task.uid,
            created_timestamp=datetime.now(),
            status=TaskStatus.IN_WORK,
        )
        await self._task_status_sender.send(task_status_log)
        video_to_text_task = self._task_data_converter.to_schema(task.input, VideoToTextTask)
        try:
            extracted_description = self._image_generator.extract_description(video_to_text_task)
        except BaseException as e:
            logger.exception(e)
            task_status_log = TaskStatusLog(
                task_uid=task.uid,
                created_timestamp=datetime.now(),
                status=TaskStatus.ERROR,
            )
            await self._task_status_sender.send(task_status_log)
        else:
            finished_timestamp = datetime.now()
            output = self._task_data_converter.to_data(task, is_input=False,
                                                       text=extracted_description.text)
            task_status_log = FullTaskStatus(
                task_uid=task.uid,
                created_timestamp=finished_timestamp,
                status=TaskStatus.FINISHED,
                output=output,
            )
            await self._task_status_sender.send(task_status_log)


class VideoDescriptionExtractor:

    def __init__(self, image_to_text: pipeline, frames_per_minute: int = 60):
        self._image_to_text = image_to_text
        self._frames_per_minute = frames_per_minute

        self._frames_per_second = self._frames_per_minute / 60
        self._dir_for_temp_frames = get_project_root().joinpath('temp_frames')
        if not self._dir_for_temp_frames.exists():
            self._dir_for_temp_frames.mkdir()

    def extract_description(self, task: VideoToTextTask) -> ExtractedDescription:
        video_path = task.internal_url
        vidcap = VideoCapture(video_path)
        fps_video = vidcap.get(cv2.CAP_PROP_FPS)
        frame_multiple_the_value = int(fps_video / self._frames_per_second)
        success, frame = vidcap.read()
        frame_count = 0
        frame_paths: List[str] = []
        video_name = os.path.basename(video_path).replace('.', '_')

        # Извлечение кадров
        while success:
            # Сохраняем кадр в директорию
            if frame_count % frame_multiple_the_value == 0:
                frame_name = f"{video_name}_frame_{frame_count}.jpg"
                frame_path = self._dir_for_temp_frames.joinpath(frame_name).as_posix()
                frame_paths.append(frame_path)
                cv2.imwrite(frame_path, frame)
            success, frame = vidcap.read()
            frame_count += 1

        # Получение описания кадров
        video_description = ""
        video_description_len = 0
        for frame_path in frame_paths:
            if video_description_len > task.text_size and task.text_size:
                break
            image_description = self._image_to_text(frame_path)
            generated_text = image_description[0]['generated_text']
            if generated_text not in video_description:
                video_description_len += len(generated_text)
                video_description += f'{generated_text}, '

        # очистка ресурсов
        for frame_path in frame_paths:
            os.remove(frame_path)
        vidcap.release()

        return ExtractedDescription(text=video_description)

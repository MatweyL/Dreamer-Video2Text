import asyncio
from asyncio import CancelledError

from transformers import pipeline

from adapters.common.converter import StrToPydantic
from adapters.common.settings import RabbitMQConnectionSettings
from adapters.inbound.rabbit.consumer import RabbitMQConsumer
from adapters.inbound.rabbit.settings import RabbitMQConsumerSettings
from adapters.outbound.rabbit.producer import RabbitMQProducer
from adapters.outbound.rabbit.settings import RabbitMQProducerSettings, TaskStatusSenderSettings
from adapters.outbound.rabbit.task_status_sender import RMQTaskStatusSender
from common.utils import build_rmq_connection_url
from domain.description_extractor import VideoDescriptionExtractor, VideoDescriptionExtractorManager
from domain.schemas.core.task import FullTask
from domain.task_data_converter import TaskDataConverter


async def main():
    rmq_settings = RabbitMQConnectionSettings()
    rmq_consumer_settings = RabbitMQConsumerSettings()
    rmq_producer_settings = RabbitMQProducerSettings()
    task_status_settings = TaskStatusSenderSettings()

    task_data_converter = TaskDataConverter()
    image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

    rmq_connection_url = build_rmq_connection_url(rmq_settings.protocol, rmq_settings.user, rmq_settings.password,
                                                  rmq_settings.host, rmq_settings.port, rmq_settings.virtual_host)
    rabbit_mq_consumer = RabbitMQConsumer(rmq_connection_url,
                                          rmq_consumer_settings.prefetch_count,
                                          rmq_consumer_settings.reconnect_timeout)
    rabbit_mq_producer = RabbitMQProducer(rmq_connection_url,
                                          rmq_producer_settings.reconnect_timeout,
                                          rmq_producer_settings.produce_retries)
    task_status_sender = RMQTaskStatusSender(rabbit_mq_producer,
                                             task_status_settings.exchange_name,
                                             task_status_settings.exchange_type,
                                             task_status_settings.routing_key)

    video_description_extractor = VideoDescriptionExtractor(image_to_text)
    video_description_extractor_manager = VideoDescriptionExtractorManager(video_description_extractor,
                                                                           task_status_sender,
                                                                           task_data_converter)
    converter = StrToPydantic(FullTask)
    await rabbit_mq_consumer.start()
    await rabbit_mq_producer.start()
    await rabbit_mq_consumer.consume_queue(rmq_consumer_settings.queue_name,
                                           video_description_extractor_manager.extract_description,
                                           converter)
    try:
        await asyncio.Future()
    except CancelledError:
        pass
    await rabbit_mq_consumer.stop()
    await rabbit_mq_producer.stop()


if __name__ == '__main__':
    asyncio.run(main())

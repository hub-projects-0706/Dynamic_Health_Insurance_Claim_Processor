import os
import json
import time
import logging
import threading
import pika
from ml_pipeline import execute_training_job

logger = logging.getLogger("claims_backend")

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "guest")

QUEUE_NAME = os.getenv("RABBITMQ_QUEUE", "training.job.queue")
EXCHANGE_NAME = os.getenv("RABBITMQ_EXCHANGE", "training.job.exchange")
ROUTING_KEY = os.getenv("RABBITMQ_ROUTING_KEY", "training.job.routingkey")


def get_connection():
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    parameters = pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        credentials=credentials,
        connection_attempts=3,
        retry_delay=2
    )
    return pika.BlockingConnection(parameters)


def publish_training_job(job_id: str, algorithm_type_str: str):
    payload = json.dumps({
        "jobId": job_id,
        "algorithmType": algorithm_type_str
    })

    try:
        logger.info("Publishing training job [%s] to RabbitMQ exchange [%s]...", job_id, EXCHANGE_NAME)
        connection = get_connection()
        channel = connection.channel()

        channel.exchange_declare(exchange=EXCHANGE_NAME, exchange_type="direct", durable=True)
        channel.queue_declare(queue=QUEUE_NAME, durable=True)
        channel.queue_bind(exchange=EXCHANGE_NAME, queue=QUEUE_NAME, routing_key=ROUTING_KEY)

        channel.basic_publish(
            exchange=EXCHANGE_NAME,
            routing_key=ROUTING_KEY,
            body=payload,
            properties=pika.BasicProperties(
                delivery_mode=2  # Make message persistent
            )
        )
        connection.close()
        logger.info("Successfully queued job [%s] to RabbitMQ queue.", job_id)
    except Exception as e:
        logger.warning(
            "RabbitMQ connection unavailable. Falling back to internal async execution for job [%s]: %s",
            job_id, e
        )
        # Fallback to local background thread execution
        thread = threading.Thread(
            target=execute_training_job,
            args=(job_id, algorithm_type_str),
            daemon=True
        )
        thread.start()


def start_rabbitmq_consumer():
    def consumer_loop():
        logger.info("Starting background RabbitMQ consumer thread for queue [%s]...", QUEUE_NAME)
        while True:
            try:
                connection = get_connection()
                channel = connection.channel()

                channel.exchange_declare(exchange=EXCHANGE_NAME, exchange_type="direct", durable=True)
                channel.queue_declare(queue=QUEUE_NAME, durable=True)
                channel.queue_bind(exchange=EXCHANGE_NAME, queue=QUEUE_NAME, routing_key=ROUTING_KEY)

                def on_message(ch, method, properties, body):
                    try:
                        data = json.loads(body)
                        job_id = data.get("jobId")
                        algo_str = data.get("algorithmType")
                        logger.info("RabbitMQ Consumer dequeued job [%s] for algorithm [%s]", job_id, algo_str)

                        execute_training_job(job_id, algo_str)
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                    except Exception as err:
                        logger.error("Failed to process dequeued training job message: %s", err)
                        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

                channel.basic_consume(queue=QUEUE_NAME, on_message_callback=on_message)
                channel.start_consuming()

            except Exception as e:
                logger.warning("RabbitMQ consumer disconnected or unavailable: %s. Retrying in 10 seconds...", e)
                time.sleep(10)

    t = threading.Thread(target=consumer_loop, daemon=True)
    t.start()

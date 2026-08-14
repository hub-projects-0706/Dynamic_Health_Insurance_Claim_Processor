import sys
import os
import json
import pika

# Ensure src root is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.services.decision_engine import evaluate_claim
from src.database.db import SessionLocal, save_processed_claim_record

RABBITMQ_HOST = os.environ.get('RABBITMQ_HOST', 'localhost')
RABBITMQ_PORT = int(os.environ.get('RABBITMQ_PORT', 5672))
RABBITMQ_USER = os.environ.get('RABBITMQ_USER', 'guest')
RABBITMQ_PASS = os.environ.get('RABBITMQ_PASS', 'guest')

QUEUE_NAME = 'claims.feature.queue'


def callback(ch, method, properties, body):
    """
    Consumer callback handling incoming claim payload messages.
    """
    try:
        payload = json.loads(body.decode('utf-8'))
        print(f"[RABBITMQ CONSUMER] Received Claim [{payload.get('claim_id')}] for feature engineering & ML adjudication...")

        # Process through Decision Engine
        evaluation_result = evaluate_claim(payload)

        # Persist to Database
        db = SessionLocal()
        try:
            save_processed_claim_record(db, payload, evaluation_result)
        finally:
            db.close()

        print(f"[RABBITMQ CONSUMER] Successfully Adjudicated Claim [{payload.get('claim_id')}] -> Route: {evaluation_result['route']}")
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        print(f"[RABBITMQ CONSUMER ERROR] Processing failed: {e}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def start_consumer():
    """
    Starts the RabbitMQ consumer listening loop.
    """
    try:
        credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
        parameters = pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            credentials=credentials
        )
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()

        channel.queue_declare(queue=QUEUE_NAME, durable=True)
        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback)

        print(f"[RABBITMQ CONSUMER] Listening for messages on '{QUEUE_NAME}'... Press CTRL+C to exit.")
        channel.start_consuming()
    except Exception as e:
        print(f"[RABBITMQ WARN] Consumer launch warning: {e}")


if __name__ == '__main__':
    start_consumer()

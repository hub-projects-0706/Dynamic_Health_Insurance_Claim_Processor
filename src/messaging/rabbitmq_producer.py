import sys
import os
import json
import pika

# Ensure src root is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

RABBITMQ_HOST = os.environ.get('RABBITMQ_HOST', 'localhost')
RABBITMQ_PORT = int(os.environ.get('RABBITMQ_PORT', 5672))
RABBITMQ_USER = os.environ.get('RABBITMQ_USER', 'guest')
RABBITMQ_PASS = os.environ.get('RABBITMQ_PASS', 'guest')

EXCHANGE_NAME = 'claims.exchange'
ROUTING_KEY = 'claims.process'
QUEUE_NAME = 'claims.feature.queue'
DLQ_NAME = 'claims.dlq'


def publish_claim_event(claim_payload: dict) -> bool:
    """
    Publishes raw claim payload to RabbitMQ message queue.
    """
    try:
        credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
        parameters = pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            credentials=credentials,
            connection_attempts=1,
            retry_delay=1
        )
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()

        # Declare Direct Exchange
        channel.exchange_declare(exchange=EXCHANGE_NAME, exchange_type='direct', durable=True)

        # Declare Queue with Dead Letter Exchange (DLX) fallback
        args = {
            'x-dead-letter-exchange': '',
            'x-dead-letter-routing-key': DLQ_NAME
        }
        channel.queue_declare(queue=QUEUE_NAME, durable=True, arguments=args)
        channel.queue_declare(queue=DLQ_NAME, durable=True)
        channel.queue_bind(exchange=EXCHANGE_NAME, queue=QUEUE_NAME, routing_key=ROUTING_KEY)

        # Publish Message
        message_body = json.dumps(claim_payload)
        channel.basic_publish(
            exchange=EXCHANGE_NAME,
            routing_key=ROUTING_KEY,
            body=message_body,
            properties=pika.BasicProperties(
                delivery_mode=2,  # Make message persistent
                content_type='application/json'
            )
        )

        print(f"[RABBITMQ] Published Claim [{claim_payload.get('claim_id')}] to {EXCHANGE_NAME}")
        connection.close()
        return True
    except Exception as e:
        print(f"[RABBITMQ WARN] Message queue offline or unavailable ({e}). Falling back to synchronous processing.")
        return False


if __name__ == '__main__':
    sample = {
        'claim_id': 'CLM-RABBIT-01',
        'policy_id': 'POL-50112',
        'patient_id': 'PAT-8012',
        'provider_id': 'PRV-105',
        'policy_status': 'ACTIVE',
        'icd10_diagnosis_code': 'J06.9',
        'cpt_procedure_code': '99213',
        'code_mismatch_score': 0.05,
        'claimed_amount': 125.00,
        'regional_benchmark_cost': 120.00,
        'provider_sanction_flag': 0,
        'is_duplicate_claim': 0,
        'prior_claim_count_30d': 1
    }
    publish_claim_event(sample)

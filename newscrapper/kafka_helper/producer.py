from kafka import KafkaProducer
import json

class KafkaJSONProducer:
    def __init__(self, brokers):
        self.producer = KafkaProducer(bootstrap_servers=brokers)

    def send_json(self, topic, message):
        serialized_message = json.dumps(message).encode('utf-8')
        self.producer.send(topic, value=serialized_message)
        self.producer.flush()
        print("[kafka][KafkaJSONProducer][send_json] success")

    def close(self):
        self.producer.close()
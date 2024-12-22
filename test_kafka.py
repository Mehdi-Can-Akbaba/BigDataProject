from kafka import KafkaConsumer

# Kafka ayarları
BROKER = "localhost:9092"
INPUT_TOPIC = "input-topic"

def kafka_consumer():
    return KafkaConsumer(
        INPUT_TOPIC,
        bootstrap_servers=BROKER,
        value_deserializer=lambda m: m.decode('utf-8'),
        auto_offset_reset='earliest',
        enable_auto_commit=True
    )

def main():
    consumer = kafka_consumer()

    for message in consumer:
        print(f"Alınan mesaj: {message.value}")

if __name__ == "__main__":
    main()

import random

import joblib
import numpy as np
from kafka import KafkaConsumer, KafkaProducer

# Kafka ayarları
BROKER = "localhost:9092"
INPUT_TOPIC = "input-topic"
NORMAL_TOPIC = "normal-topic"
ANOMALY_TOPIC = "anomaly-topic"

# KMeans model dosyası
MODEL_PATH = "kmeans_model.joblib"  # joblib kullanarak kaydedilen model dosyası


def load_kmeans_model():
    print(">>> KMeans model dosyası yükleniyor...")
    model = joblib.load(MODEL_PATH)  # joblib ile modeli yükle
    print(f">>> Model başarıyla yüklendi. Kümelerin merkez sayısı: {len(model.cluster_centers_)}")
    print(">>> KMeans modeli başarıyla oluşturuldu.")
    return model


def kafka_producer():
    return KafkaProducer(
        bootstrap_servers=BROKER,
        value_serializer=lambda v: str(v).encode('utf-8')
    )


def kafka_consumer():
    return KafkaConsumer(
        INPUT_TOPIC,
        bootstrap_servers=BROKER,
        value_deserializer=lambda m: m.decode('utf-8'),
        auto_offset_reset='earliest',
        enable_auto_commit=True
    )


def classify_data(model, data):
    try:
        # KMeans modelinin predict metodu kullanılmalı
        cluster = model.predict([data])[0]
        print(f">>> Veri kümesi sınıflandırılıyor. En yakın küme: {cluster}")
        return cluster
    except Exception as e:
        print(f">>> Veri sınıflandırılamadı. Hata: {str(e)}")
        return None


def get_random_samples(data, num_samples=1000):
    # Başından, ortasından ve sonundan rastgele örnekler al
    if len(data) < num_samples:
        print(f"Veri sayısı yetersiz. Tüketilen veri sayısı: {len(data)}")
        return data  # Eğer yeterince veri yoksa, tüm veriyi döndür

    # Baş, ortada ve son kısmından veri al
    head = data[:num_samples//3]
    middle = data[num_samples//3:num_samples*2//3]
    tail = data[num_samples*2//3:num_samples]

    # Bu üç kısımdan rastgele seçim yap
    random_samples = random.sample(head, num_samples//3) + random.sample(middle, num_samples//3) + random.sample(tail, num_samples//3)
    random.shuffle(random_samples)  # Rastgele sıralamak

    return random_samples


def main():
    print(">>> Sistem başlatılıyor...")
    model = load_kmeans_model()
    producer = kafka_producer()
    consumer = kafka_consumer()

    print(">>> Veri tüketilmeye başlanıyor...")
    consumed_data = []
    for message in consumer:
        consumed_data.append(message.value)
        # Eğer 1000 veri alındıysa, işlem sonlanacak
        if len(consumed_data) >= 1000:
            break

    if len(consumed_data) < 1000:
        print(f">>> Yeterli veri yok. Toplam veri sayısı: {len(consumed_data)}")
        return

    print(f">>> Toplam {len(consumed_data)} veri tüketildi. Rastgele örnekler alınıyor...")

    random_samples = get_random_samples(consumed_data, 1000)

    print(">>> Rastgele seçilen veriler:")
    for i, sample in enumerate(random_samples[:5], 1):  # İlk 5 rastgele seçilen veri gösterilsin
        print(f"Veri {i}: {sample}")

    print(">>> Veriler işleniyor...")
    for data in random_samples:
        try:
            # Çift tırnakları temizleyerek veriyi işleyelim
            data_cleaned = data.replace('"', '')
            features = np.array([float(x) for x in data_cleaned.split(",")])
            cluster = classify_data(model, features)
            if cluster is not None:
                if cluster == 0:  # Normal cluster
                    producer.send(NORMAL_TOPIC, value=data)
                    print(f">>> Veri normal olarak sınıflandırıldı ve {NORMAL_TOPIC} başlığına gönderildi.")
                else:  # Anomaly cluster
                    producer.send(ANOMALY_TOPIC, value=data)
                    print(f">>> Veri anormal olarak sınıflandırıldı ve {ANOMALY_TOPIC} başlığına gönderildi.")
        except Exception as e:
            print(f">>> Veri işlenirken hata oluştu: {data}, Hata: {str(e)}")

    print(">>> Tüm işlemler tamamlandı.")


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# CSV dosyasını yükle
df = pd.read_csv('db.csv')

# Verileri standartlaştır
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)

# KMeans modelini oluştur ve eğit
n_clusters = 2  # Küme sayısını belirleyebilirsiniz
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

# Modeli eğit
kmeans.fit(scaled_features)

# Modeli joblib ile kaydet
joblib.dump(kmeans, 'kmeans_model.joblib')

print("KMeans modeli başarıyla eğitildi ve kmeans_model.joblib dosyasına kaydedildi.")

# Veri sınıflandırması yap
predictions = kmeans.predict(scaled_features)

# Örnek olarak, ilk 5 veriyi ve sınıflandırmalarını yazdıralım
for i in range(5):
    print(f"Veri {i+1}: {df.iloc[i].values} -> Küme: {predictions[i]}")

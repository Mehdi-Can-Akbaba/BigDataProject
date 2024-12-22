import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# CSV dosyasını yükle
df = pd.read_csv('db.csv')

# Eksik verileri kontrol etme ve doldurma
if df.isnull().sum().any():
    print("Eksik veriler tespit edildi. Dolduruluyor...")
    df.fillna(df.mean(), inplace=True)  # Eksik verileri ortalama ile doldur

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

# Silhouette Skoru ile model performansını değerlendirme
silhouette_avg = silhouette_score(scaled_features, predictions)
print(f"Silhouette Skoru: {silhouette_avg}")

# Veriyi ve küme sonuçlarını birleştir
df['Cluster'] = predictions
df['Silhouette_Score'] = silhouette_avg  # Silhouette skoru tüm veri için aynı olduğundan her satıra aynı değeri ekliyoruz

# Veri çerçevesinin ilk 5 satırını yazdıralım
print(df.head())

# Sonuçları CSV dosyasına kaydet
output_file = 'kmeans_results.csv'
df.to_csv(output_file, index=False)

# CSV kaydettikten sonra bilgi ver
print(f"Kümeleme sonuçları ve metrikler {output_file} dosyasına kaydedildi.")

# Korelasyon matrisi görselleştirmesi
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Korelasyon Matrisi')
plt.show()

# Kümeleme sonuçlarının görselleştirilmesi (scatter plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue='Cluster', data=df, palette='viridis', s=100)
plt.title('KMeans Kümeleme Sonuçları')
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.legend(title='Küme', loc='upper right')
plt.show()

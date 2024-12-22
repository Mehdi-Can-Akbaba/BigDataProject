from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans

# Spark oturumunu başlat
spark = SparkSession.builder.appName("SparkKMeansModelTraining").getOrCreate()

# CSV dosyasını yükle
input_path = "db.csv"  # CSV dosya yolunu buraya ekleyin
df = spark.read.csv(input_path, header=True, inferSchema=True)

# Özellikleri birleştir
feature_columns = df.columns
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df_assembled = assembler.transform(df)

# Verileri standartlaştır
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)
scaler_model = scaler.fit(df_assembled)
df_scaled = scaler_model.transform(df_assembled)

# KMeans modelini eğit
kmeans = KMeans(featuresCol="scaled_features", k=2, seed=42)
kmeans_model = kmeans.fit(df_scaled)

# Modeli kaydet
model_path = "spark_kmeans_model"
kmeans_model.write().overwrite().save(model_path)

print(f"KMeans modeli başarıyla {model_path} konumuna kaydedildi.")

spark.stop()

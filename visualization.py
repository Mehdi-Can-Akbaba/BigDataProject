import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import zscore

# Veri setini yükleme
file_path = 'MiningProcess_Flotation_Plant_Database.csv'  # Dosya yolunuza göre güncelleyin
data = pd.read_csv(file_path)

# Eksik değerlerin toplamını kontrol etme
missing_values = data.isnull().sum()

# Eksik veri içeren sütunları filtreleme
missing_data = missing_values[missing_values > 0]

# Sonuçları yazdırma
if missing_data.empty:
    print("Veri setinde eksik veri bulunmamaktadır.")
else:
    print("Eksik veri bulunan sütunlar ve eksik değer sayıları:")
    print(missing_data)


# Veri setini yükleme
data = pd.read_csv(file_path)
output_folder = 'output_folder'

# Boxplot
def generate_boxplots(data, output_folder):
    # Boxplot klasörünü oluşturma
    boxplot_folder = os.path.join(output_folder, 'Boxplot')
    os.makedirs(boxplot_folder, exist_ok=True)

    for column_index in range(1, 24):  # 2. sütun (index 1) ile 24. sütun (index 23) arasında
        column_name = data.columns[column_index]

        # Virgülleri noktaya çevirme
        data[column_name] = data[column_name].replace({',': '.'}, regex=True)

        # Sayıya dönüştürme
        data[column_name] = pd.to_numeric(data[column_name], errors='coerce')

        # Sayısal verilerin doğru şekilde dönüştürülüp dönüştürülmediğini kontrol etme
        print(f"'{column_name}' sütunundaki veri tipi: {data[column_name].dtype}")

        # Sadece sayısal veriler üzerinde işlem yapmak için
        if data[column_name].dtype in ['int64', 'float64']:
            # NaN değerlerini temizle
            data_clean = data[column_name].dropna()

            # Z-score hesaplama
            z_scores = zscore(data_clean)

            # Aykırı değerlerin tespiti (mutlak Z-skoru 3'ten büyük olanlar)
            outliers = data_clean[abs(z_scores) > 3]

            # Aykırı değerlerin sayısı
            outlier_count = len(outliers)

            # Sonuçları yazdırma
            print(f"'{column_name}' sütununda {outlier_count} aykırı değer bulundu.")

            # Boxplot çizimi
            plt.figure(figsize=(6, 10))
            sns.boxplot(y=data_clean, color='indianred', width=0.5)

            # Her kutu üzerindeki değerleri gösterme
            mean_value = data_clean.mean()
            median_value = data_clean.median()
            plt.text(mean_value, 0, f"Mean: {mean_value:.2f}", color="red", ha="center", va="bottom", rotation=0)
            plt.text(median_value, 0, f"Median: {median_value:.2f}", color="blue", ha="center", va="bottom", rotation=0)

            # Başlık ve eksen etiketlerini ekleme
            plt.title(f"Boxplot of {column_name}", fontsize=16)
            plt.ylabel(column_name, fontsize=12)
            plt.xlabel("Values", fontsize=12)

            # Grafik kaydetme
            file_name = f"{column_name}_boxplot.png"
            plt.savefig(os.path.join(boxplot_folder, file_name), dpi=300)
            plt.close()
            print(f"{column_name} için boxplot grafiği kaydedildi.")
        else:
            print(f"'{column_name}' sütunu sayısal veri içermiyor ve atlandı.")


# Histogram Grafik
def generate_histograms(data, output_folder):
    # Histogram_Graphs klasörünü oluşturma
    histogram_graphs_folder = os.path.join(output_folder, 'Histogram_Graphs')
    os.makedirs(histogram_graphs_folder, exist_ok=True)

    for column_index in range(1, 24):  # 2. sütun (index 1) ile 24. sütun (index 23) arasında
        column_name = data.columns[column_index]

        # Virgülleri noktaya çevirme
        data[column_name] = data[column_name].replace({',': '.'}, regex=True)  # Virgülleri noktaya çevir

        # Sayıya dönüştürme
        data[column_name] = pd.to_numeric(data[column_name], errors='coerce')  # Sayıya dönüştürme

        # Sayısal verilerin doğru şekilde dönüştürülüp dönüştürülmediğini kontrol etme
        print(f"'{column_name}' sütunundaki veri tipi: {data[column_name].dtype}")

        # Sadece sayısal veriler üzerinde işlem yapmak için
        if data[column_name].dtype in ['int64', 'float64']:
            # NaN değerlerini temizle
            data_clean = data[column_name].dropna()

            # Z-score hesaplama
            z_scores = zscore(data_clean)

            # Aykırı değerlerin tespiti (mutlak Z-skoru 3'ten büyük olanlar)
            outliers = data_clean[abs(z_scores) > 3]

            # Aykırı değerlerin sayısı
            outlier_count = len(outliers)

            # Sonuçları yazdırma
            print(f"'{column_name}' sütununda {outlier_count} aykırı değer bulundu.")

            # Histogram çizimi
            plt.figure(figsize=(10, 6))
            sns.histplot(data_clean, kde=False, bins=20, color='powderblue')  # Histogram çizimi
            plt.title(f"Histogram of {column_name}", fontsize=16)
            plt.xlabel(column_name, fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.grid(True)

            # Grafik kaydetme
            file_name = f"{column_name}_histogram.png"
            plt.savefig(os.path.join(histogram_graphs_folder, file_name), dpi=300)
            plt.close()
            print(f"{column_name} için histogram grafiği kaydedildi.")
        else:
            print(f"'{column_name}' sütunu sayısal veri içermiyor ve atlandı.")


# Yoğunluk (KDE) Grafikleri
def generate_kde_plots(data, output_folder):
    # KDE Graphs klasörünü oluşturma
    kde_graphs_folder = os.path.join(output_folder, 'KDE_Graphs')
    os.makedirs(kde_graphs_folder, exist_ok=True)

    for column in data.columns:
        # Sadece sayısal sütunlarla çalışmak için
        if data[column].dtype == 'object':
            # Virgülleri noktaya çevirme ve sayıya dönüştürme
            data[column] = data[column].replace({',': '.'}, regex=True)
            data[column] = pd.to_numeric(data[column], errors='coerce')

        # Sayısal sütunları kontrol etme
        if data[column].dtype in ['float64', 'int64']:
            plt.figure(figsize=(10, 6))
            sns.kdeplot(data[column].dropna(), shade=True)
            plt.title(f"KDE Plot of {column}", fontsize=16)
            plt.xlabel(column, fontsize=12)
            plt.ylabel("Density", fontsize=12)
            plt.grid(True)

            # Grafik kaydetme
            file_name = f"{column}_kde.png"
            plt.savefig(os.path.join(kde_graphs_folder, file_name), dpi=300)
            plt.close()
            print(f"{column} için KDE grafiği kaydedildi.")
        else:
            print(f"{column} sütunu sayısal veri içermiyor ve atlandı.")


# Gruplandırılmış Yoğunluk (KDE) Grafikleri
def generate_groups_kde_plots(data, output_folder):
    # KDE Graphs klasörünü oluşturma
    kde_graphs_folder = os.path.join(output_folder, 'KDE_Graphs')
    os.makedirs(kde_graphs_folder, exist_ok=True)

    # 23. ve 24. sütunları kontrol
    column_23 = data.columns[22]
    column_24 = data.columns[23]

    # Virgülleri noktaya çevirme ve sayıya dönüştürme
    data[column_23] = data[column_23].replace({',': '.'}, regex=True)
    data[column_23] = pd.to_numeric(data[column_23], errors='coerce')

    data[column_24] = data[column_24].replace({',': '.'}, regex=True)
    data[column_24] = pd.to_numeric(data[column_24], errors='coerce')

    # Sadece sayısal veriler üzerinde işlem yapmak için
    if data[column_23].dtype in ['float64', 'int64'] and data[column_24].dtype in ['float64', 'int64']:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data[column_23].dropna(), shade=True, label=column_23, color="blue")
        sns.kdeplot(data[column_24].dropna(), shade=True, label=column_24, color="green")
        plt.title(f"KDE Plot of {column_23} and {column_24}", fontsize=16)
        plt.xlabel("Values", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend()
        plt.grid(True)

        # Grafik kaydetme
        file_name = f"{column_23}_{column_24}_kde.png"
        plt.savefig(os.path.join(kde_graphs_folder, file_name), dpi=300)
        plt.close()
        print(f"{column_23} ve {column_24} için KDE grafiği kaydedildi.")
    else:
        print(f"23. veya 24. sütun sayısal veri içermiyor ve atlandı.")

    # 16-22. sütunlar için KDE çizimi
    columns_16_to_22 = data.columns[15:22]  # 16. sütundan (index 15) 22. sütuna (index 21)

    for column in columns_16_to_22:
        data[column] = data[column].replace({',': '.'}, regex=True)
        data[column] = pd.to_numeric(data[column], errors='coerce')

    # Sütunlar arasında sayısal olmayanları filtreleme
    numeric_data = data[columns_16_to_22].dropna()

    plt.figure(figsize=(12, 8))
    for column in numeric_data.columns:
        sns.kdeplot(numeric_data[column].dropna(), shade=True, label=column)
    plt.title("KDE Plot of Columns 16 to 22", fontsize=16)
    plt.xlabel("Values", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True)

    # Grafik kaydetme
    file_name = "columns_16_to_22_kde.png"
    plt.savefig(os.path.join(kde_graphs_folder, file_name), dpi=300)
    plt.close()
    print("16-22 sütunları için KDE grafiği kaydedildi.")

    # 2. ve 3. sütunlar için KDE çizimi
    column_2 = data.columns[1]
    column_3 = data.columns[2]

    data[column_2] = data[column_2].replace({',': '.'}, regex=True)
    data[column_2] = pd.to_numeric(data[column_2], errors='coerce')

    data[column_3] = data[column_3].replace({',': '.'}, regex=True)
    data[column_3] = pd.to_numeric(data[column_3], errors='coerce')

    if data[column_2].dtype in ['float64', 'int64'] and data[column_3].dtype in ['float64', 'int64']:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data[column_2].dropna(), shade=True, label=column_2, color="purple")
        sns.kdeplot(data[column_3].dropna(), shade=True, label=column_3, color="orange")
        plt.title(f"KDE Plot of {column_2} and {column_3}", fontsize=16)
        plt.xlabel("Values", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend()
        plt.grid(True)

        file_name = f"{column_2}_{column_3}_kde.png"
        plt.savefig(os.path.join(kde_graphs_folder, file_name), dpi=300)
        plt.close()
        print(f"{column_2} ve {column_3} için KDE grafiği kaydedildi.")

    # 4., 5. ve 6. sütunlar için KDE çizimi
    columns_4_to_6 = data.columns[3:6]

    for column in columns_4_to_6:
        data[column] = data[column].replace({',': '.'}, regex=True)
        data[column] = pd.to_numeric(data[column], errors='coerce')

    numeric_data_4_to_6 = data[columns_4_to_6].dropna()

    plt.figure(figsize=(10, 6))
    for column in numeric_data_4_to_6.columns:
        sns.kdeplot(numeric_data_4_to_6[column].dropna(), shade=True, label=column)
    plt.title("KDE Plot of Columns 4 to 6", fontsize=16)
    plt.xlabel("Values", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True)

    file_name = "columns_4_to_6_kde.png"
    plt.savefig(os.path.join(kde_graphs_folder, file_name), dpi=300)
    plt.close()
    print("4-6 sütunları için KDE grafiği kaydedildi.")

    # 9., 10., 11., 12., 13., 14. ve 15. sütunlar için KDE çizimi
    columns_9_to_15 = data.columns[8:15]

    for column in columns_9_to_15:
        data[column] = data[column].replace({',': '.'}, regex=True)
        data[column] = pd.to_numeric(data[column], errors='coerce')

    numeric_data_9_to_15 = data[columns_9_to_15].dropna()

    plt.figure(figsize=(12, 8))
    for column in numeric_data_9_to_15.columns:
        sns.kdeplot(numeric_data_9_to_15[column].dropna(), shade=True, label=column)
    plt.title("KDE Plot of Columns 9 to 15", fontsize=16)
    plt.xlabel("Values", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True)

    file_name = "columns_9_to_15_kde.png"
    plt.savefig(os.path.join(kde_graphs_folder, file_name), dpi=300)
    plt.close()
    print("9-15 sütunları için KDE grafiği kaydedildi.")

# Heatmap
def generate_heatmaps(data, output_folder):
    # Heatmap klasörünü oluşturma
    heatmap_folder = os.path.join(output_folder, 'Heatmap')
    os.makedirs(heatmap_folder, exist_ok=True)

    for column_index in range(1, 24):  # 2. sütun (index 1) ile 24. sütun (index 23) arasında
        column_name = data.columns[column_index]

        # Virgülleri noktaya çevirme
        data[column_name] = data[column_name].replace({',': '.'}, regex=True)

        # Sayıya dönüştürme
        data[column_name] = pd.to_numeric(data[column_name], errors='coerce')

        # Sayısal verilerin doğru şekilde dönüştürülüp dönüştürülmediğini kontrol etme
        print(f"'{column_name}' sütunundaki veri tipi: {data[column_name].dtype}")

        # Sadece sayısal veriler üzerinde işlem yapmak için
        if data[column_name].dtype in ['int64', 'float64']:
            # NaN değerlerini temizle
            data_clean = data[column_name].dropna()

            # Heatmap çizimi
            plt.figure(figsize=(10, 8))
            sns.heatmap(data_clean.values.reshape(-1, 1), cmap="coolwarm", cbar=True, annot=False, yticklabels=False)

            # Başlık ve eksen etiketlerini ekleme
            plt.title(f"Heatmap of {column_name}", fontsize=16)
            plt.xlabel("Index", fontsize=12)
            plt.ylabel(column_name, fontsize=12)

            # Grafik kaydetme
            file_name = f"{column_name}_heatmap.png"
            plt.savefig(os.path.join(heatmap_folder, file_name), dpi=300)
            plt.close()
            print(f"{column_name} için heatmap grafiği kaydedildi.")
        else:
            print(f"'{column_name}' sütunu sayısal veri içermiyor ve atlandı.")



generate_groups_kde_plots(data, output_folder)
generate_kde_plots(data, output_folder)
generate_boxplots(data,output_folder)
generate_histograms(data, output_folder)
generate_heatmaps(data, output_folder)




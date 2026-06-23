from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Inisialisasi Klaster 24 Cores terikat IP Tailscale Lu
spark = SparkSession.builder \
    .appName("FraudDetection_Phase1_Preprocessing") \
    .master("spark://100.86.108.61:7077") \
    .config("spark.driver.host", "100.86.108.61") \
    .config("spark.driver.bindAddress", "100.86.108.61") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("\n" + "="*75)
print(" FASE 1: INGESTION DATA HDFS FULL & ANALISIS KORELASI BIG DATA")
print("="*75)

# 1. Load Data FULL dari HDFS Tailscale (Memenuhi syarat HDFS & Judul Artikel)
path_hdfs = "hdfs://100.86.108.61:9000/user/kelompok2/fraud_data/creditcard_2023.csv"
df_raw = spark.read.csv(path_hdfs, header=True, inferSchema=True)

# Cukup caching data sekali di memori agar eksekusi di bawahnya(biar cepet ae)
df_raw.cache()

# 2. Hitung Distribusi Transaksi
print("[PROSES] Menghitung Distribusi Kelas Transaksi...")
distribusi = df_raw.groupBy("Class").count().collect()
total_data = 0

for row in distribusi:
    status = "Fraud (Class 1)" if row['Class'] == 1 else "Normal (Class 0)"
    print(f"   - {status}: {row['count']} transaksi")
    total_data += row['count']

print(f"[INFO] Total data skala besar yang diproses: {total_data} baris.")

# 3. Analisis Korelasi Fitur Utama terhadap Class
print("[PROSES] Menghitung Matriks Korelasi Fitur Utama terhadap Class...")
fitur_penting = ["V1", "V2", "V3", "V4", "Amount"]
for fitur in fitur_penting:
    korelasi = df_raw.stat.corr(fitur, "Class")
    print(f"   - Nilai Korelasi [{fitur} vs Class]: {korelasi:.4f}")

# 4. Simpan Output Bersih ke HDFS sebagai Parquet untuk Bahan Skrip Analisis Lanjutan
print("[PROSES] Menyimpan Data Hasil Preprocessing ke HDFS...")
df_raw.write.mode("overwrite").parquet("hdfs://100.86.108.61:9000/user/kelompok2/fraud_data/cleaned_data.parquet")

print("[SUKSES] Skrip 1 Selesai Sempurna di Klaster 24 Cores!")
print("="*75 + "\n")

spark.stop()

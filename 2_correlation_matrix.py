from pyspark.sql import SparkSession
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
import os

# 1. Inisialisasi Klaster 24 Cores Terdistribusi
spark = SparkSession.builder \
    .appName("FraudDetection_Phase2_CorrelationMatrix") \
    .master("spark://100.86.108.61:7077") \
    .config("spark.driver.host", "100.86.108.61") \
    .config("spark.driver.bindAddress", "100.86.108.61") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("\n" + "="*75)
print("       FASE 2: PERHITUNGAN MATRIKS KORELASI GLOBAL (KLASTER 24 CORES)")
print("="*75)

# 2. Load Data Parquet Hasil Preprocessing dari HDFS (Sangat Cepat & Ringan)
path_parquet = "hdfs://100.86.108.61:9000/user/kelompok2/fraud_data/cleaned_data.parquet"
df = spark.read.parquet(path_parquet)

# Ambil semua kolom fitur numerik (V1 sampai V28 + Amount + Class)
# Kita drop kolom 'id' jika ada agar tidak merusak korelasi
all_columns = [c for c in df.columns if c not in ['id']]

print(f"[INFO] Memproses matriks korelasi untuk {len(all_columns)} fitur terdistribusi...")

# 3. Transformasi Data menjadi Vektor untuk Spark MLlib
assembler = VectorAssembler(inputCols=all_columns, outputCol="features")
df_vector = assembler.transform(df).select("features")

# 4. Hitung Matriks Korelasi Pearson secara Paralel di 24 Cores
print("[PROSES] Menghitung Pearson Correlation Matrix (Beban Kerja Paralel)...")
matrix = Correlation.corr(df_vector, "features").head()
correlation_matrix = matrix[0].toArray()

# 5. Konversi Hasil Matriks ke DataFrame Spark untuk Diekspor
print("[PROSES] Menformat hasil matriks korelasi...")
output_data = []
for i, col_name_i in enumerate(all_columns):
    row_dict = {"Fitur": col_name_i}
    for j, col_name_j in enumerate(all_columns):
        row_dict[col_name_j] = float(correlation_matrix[i][j])
    output_data.append(row_dict)

df_corr_result = spark.createDataFrame(output_data)

# 6. Ekspor Hasil Korelasi Menjadi 1 File CSV Tunggal ke HDFS
# Menggunakan .coalesce(1) karena ukuran matriks kecil (30x30), aman tanpa beban
path_output_csv = "hdfs://100.86.108.61:9000/user/kelompok2/fraud_data/matriks_korelasi_final"
df_corr_result.coalesce(1).write.mode("overwrite").option("header", "true").csv(path_output_csv)

print("\n[SUKSES] Perhitungan Matriks Korelasi Selesai Sempurna!")
print(f"[HDFS] Hasil disimpan di: {path_output_csv}")
print("="*75 + "\n")

spark.stop()

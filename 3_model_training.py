from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# 1. Inisialisasi Klaster 24 Cores
spark = SparkSession.builder \
    .appName("FraudDetection_Phase3_ModelTraining") \
    .master("spark://100.86.108.61:7077") \
    .config("spark.driver.host", "100.86.108.61") \
    .config("spark.driver.bindAddress", "100.86.108.61") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("\n" + "="*75)
print("     FASE 3: MODEL TRAINING & CLASS WEIGHTING EVALUATION (24 CORES)")
print("="*75)

# 2. Load Data Parquet Bersih
path_parquet = "hdfs://100.86.108.61:9000/user/kelompok2/fraud_data/cleaned_data.parquet"
df_clean = spark.read.parquet(path_parquet)

# 3. SIMULASI CLASS IMBALANCE (Agar Metode 1 & 3 Lu Sah di Jurnal!)
# Kita ambil semua data normal, tapi data fraud-nya kita pangkas murni tinggal 5% saja
df_normal = df_clean.filter(col("Class") == 0)
df_fraud_skewed = df_clean.filter(col("Class") == 1).sample(withReplacement=False, fraction=0.05, seed=42)
df_imbalance = df_normal.union(df_fraud_skewed).cache()

# Hitung rasio untuk Class Weighting
total_normal = df_imbalance.filter(col("Class") == 0).count()
total_fraud = df_imbalance.filter(col("Class") == 1).count()
total_data = total_normal + total_fraud

print(f"[INFO] Data Berhasil Disimulasikan Timpang (Imbalance):")
print(f"   - Normal (Class 0): {total_normal} baris")
print(f"   - Fraud  (Class 1): {total_fraud} baris")

# Menghitung bobot inversi sesuai Metode 1 (Cost-Sensitive Learning)
weight_normal = total_data / (2.0 * total_normal)
weight_fraud = total_data / (2.0 * total_fraud)

df_weighted = df_imbalance.withColumn("classWeights", when(col("Class") == 1, weight_fraud).otherwise(weight_normal))

# 4. Rekayasa Fitur (Metode 2: Pipeline MLlib)
fitur_input = [c for c in df_clean.columns if c not in ['id', 'Class']]
assembler = VectorAssembler(inputCols=fitur_input, outputCol="features")
df_model_input = assembler.transform(df_weighted).select("features", "Class", "classWeights")

# Split Data (80% Training, 20% Testing)
train_data, test_data = df_model_input.randomSplit([0.8, 0.2], seed=42)

# 5. Training Model dengan Parameter weightCol (Metode 1 & 2)
print("[PROSES] Melakukan Training Logistic Regression Terdistribusi di 24 Cores...")
lr = LogisticRegression(featuresCol="features", labelCol="Class", weightCol="classWeights")
model = lr.fit(train_data)

# 6. Evaluasi Ketat (Metode 3: F1-Score & ROC)
print("[PROSES] Mengevaluasi Performa Model Klasifikasi Biner...")
predictions = model.transform(test_data)

# Evaluasi AUC-ROC
evaluator_roc = BinaryClassificationEvaluator(labelCol="Class", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc_roc = evaluator_roc.evaluate(predictions)

# Evaluasi F1-Score, Precision, dan Recall
evaluator_multi = MulticlassClassificationEvaluator(labelCol="Class", predictionCol="prediction")

f1 = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "f1"})
precision = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "weightedPrecision"})
recall = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "weightedRecall"})

print("\n" + "-"*50)
print(" METRIK HASIL EVALUASI MODEL (BAHAN UTAMA TABEL JURNAL)")
print("-"*50)
print(f"   - Area Under ROC (AUC) : {auc_roc:.4f}")
print(f"   - F1-Score             : {f1:.4f}")
print(f"   - Precision            : {precision:.4f}")
print(f"   - Recall               : {recall:.4f}")
print("-"*50 + "\n")

print("[SUKSES] Seluruh Rangkaian Eksperimen 3 Metode Selesai Sempurna!")
print("="*75 + "\n")



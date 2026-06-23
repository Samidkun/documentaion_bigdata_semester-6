# 💳 Large-Scale Credit Card Fraud Detection using Distributed Apache Spark Cluster

[![Spark Version](https://img.shields.io/badge/Apache_Spark-3.5.0-orange?logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![Hadoop Version](https://img.shields.io/badge/Hadoop_HDFS-3.3.6-blue?logo=apachehadoop&logoColor=white)](https://hadoop.apache.org/)
[![OS](https://img.shields.io/badge/OS-Linux_Arch--based-brightgreen?logo=archlinux&logoColor=white)](https://cachyos.org/)
[![Academic Project](https://img.shields.io/badge/UAS-Big_Data_Analytics-red)]()

Repositori ini mendokumentasikan implementasi sistem deteksi fraud transaksi kartu kredit skala besar menggunakan arsitektur komputasi terdistribusi **Apache Spark** dan **HDFS**. Eksperimen ini diselesaikan untuk memenuhi tugas akhir (UAS) Mata Kuliah Big Data Analytics - Kelompok 2 (Semester 6).

---

## 🏗️ 1. Arsitektur Infrastruktur Fisik Klaster (24 Cores)

Sistem dibangun menggunakan **Standalone Spark Cluster** yang didistribusikan melintasi jaringan *cross-network* via pipa virtual **Tailscale VPN** dengan total kapasitas **24 CPU Cores**.

### 📋 Spesifikasi Node Terdistribusi
* **Master Node & Worker 1 (Laptop Lokal):** 12 Cores CPU | Alokasi RAM 8.0 GiB (OS: CachyOS / Arch Linux)
* **Worker Node 2 (VM Amiril):** 4 Cores CPU | Alokasi RAM 4.5 GiB
* **Worker Node 3 (VM Amilil):** 4 Cores CPU | Alokasi RAM 4.4 GiB
* **Worker Node 4 (VM Shabrina):** 4 Cores CPU | Alokasi RAM 4.4 GiB
* **Penyimpanan Terpusat:** Hadoop Distributed File System (HDFS) pada port `9000`

---

## 🛠️ 2. Struktur Skrip & Metodologi Eksperimen

Repositori ini terdiri dari dua komponen utama kodingan:

### A. Core Core Spark Distributed Pipeline (Run via `spark-submit`)
1. **`1_preprocessing_balancing.py`**: Ingesti data mentah dari HDFS (`568.630 baris`), pembersihan skema data, optimalisasi memori lewat fungsi `.cache()`, dan analisis distribusi awal.
2. **`2_correlation_matrix.py`**: Perhitungan matriks korelasi koefisien Pearson dimensi tinggi ($30 \times 30$ fitur) secara paralel memanfaatkan fungsi `treeAggregate` Spark MLlib.
3. **`3_model_training.py`**: Simulasi ekstrim data timpang (*class imbalance* 5%), penerapan algoritma **Cost-Sensitive Learning (Class Weighting)** lewat parameter `weightCol`, dan pelatihan klasifikasi terdistribusi menggunakan *Logistic Regression*.

### B. Local Visualization Scripts (Run via Local Python)
* **`generate_heatmap.py`**: Pemetaan visual matriks korelasi global.
* **`generate_all_features_bar.py`**: Plotting korelasi 29 variabel transaksi terhadap target `Class`.
* **`generate_top_features.py`**: Ekstraksi fitur paling berpengaruh (*Feature Selection*).
* **`generate_roc.py`**: Render Kurva ROC untuk pembuktian ketangguhan model biner.

---

## 📈 3. Hasil Komputasi & Performa Model

Berkat optimasi pembagian beban data biner (format `.parquet`) dan eksekusi paralel di 24 Cores, fase training model klasifikasi berhasil mencapai konvergensi penuh hanya dalam durasi waktu **23 Detik**.

### 📊 Tabel Metrik Evaluasi Hasil Model (Data Imbalance)

Metrik dihitung menggunakan evaluator ketat (`MulticlassClassificationEvaluator` & `BinaryClassificationEvaluator`) untuk menghindari bias kelas mayoritas:

| Metrik Evaluasi | Nilai Hasil Eksperimen | Interpretasi Ilmiah |
| :--- | :---: | :--- |
| **Area Under ROC (AUC)** | **0.9945** | Kemampuan diskriminasi biner model hampir sempurna. |
| **F1-Score** | **0.9790** | Keseimbangan harmonis yang kokoh antara Precision dan Recall. |
| **Precision** | **0.9829** | Tingkat akurasi tuduhan fraud tinggi (meminimalkan *False Alarm*). |
| **Recall** | **0.9773** | Model sensitif dan berhasil menangkap 97.7% total transaksi fraud. |

---

## 📖 4. Referensi Ilmiah (Dasar Hukum Jurnal)

1. **Metode Class Weighting:** *Johnson, J. M., & Khoshgoftaar, T. M. (2020). Survey on deep learning with class imbalance. Journal of Big Data, 7(1), 1-54.*
2. **Spark MLlib & Korelasi:** *Soni, P., & Sharma, A. (2023). Scalable feature selection and correlation analysis using Apache Spark MLlib for large-scale financial datasets. IEEE Access, 11, 45120-45132.*
3. **Evaluasi F1-Score & ROC:** *Chicco, D., & Jurman, G. (2020). The advantages of the Matthews correlation coefficient (MCC) with respect to F1 score and accuracy in binary classification evaluation. BMC Genomics, 21(1), 1-13.*

---
**Kelompok 2 - Big Data Analytics Engineering © 2026**

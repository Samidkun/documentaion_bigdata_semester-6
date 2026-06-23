import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("[PROSES] Membaca data matriks korelasi 18 KB...")
# Baca file csv hasil download dari HDFS tadi
df = pd.read_csv('/home/samid/matriks_korelasi.csv')

# Jadikan kolom 'Fitur' sebagai index agar matriks berbentuk persegi angka murni
df.set_index('Fitur', inplace=True)

# Atur ukuran kanvas grafik agar fitur V1-V28 terbaca jelas gak tumpang tindih
plt.figure(figsize=(20, 16))

print("[PROSES] Membuat grafik Heatmap Korelasi Pearson...")
# Buat heatmap dengan warna coolwarm (biru ke merah)
sns.heatmap(df, annot=False, cmap='coolwarm', linewidths=0.5, vmin=-1, vmax=1)

plt.title('Matriks Korelasi Fitur Transaksi Fraud - Kelompok 2', fontsize=18, pad=20)
plt.xlabel('Fitur', fontsize=12)
plt.ylabel('Fitur', fontsize=12)

# Simpan hasilnya jadi gambar PNG resolusi tinggi (HD) untuk jurnal
output_image = '/home/samid/heatmap_korelasi_fraud.png'
plt.savefig(output_image, dpi=300, bbox_inches='tight')

print(f"[SUKSES] Gambar Heatmap berhasil disimpan di: {output_image}")

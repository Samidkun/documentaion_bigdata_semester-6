import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("[PROSES] Membaca matriks korelasi global 18 KB...")
df = pd.read_csv('/home/samid/matriks_korelasi.csv')
df.set_index('Fitur', inplace=True)

# Ekstrak nilai korelasi seluruh fitur hanya terhadap variabel 'Class'
# Kita drop baris 'Class' itu sendiri agar nilainya tidak 1.0
class_correlation = df['Class'].drop('Class')

# Urutkan dari nilai korelasi tertinggi ke terendah agar grafik rapi
class_correlation = class_correlation.sort_values(ascending=False)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(15, 6))

# Warna merah untuk korelasi positif, biru untuk korelasi negatif
colors = ['#e04a4a' if x > 0 else '#4a90e2' for x in class_correlation.values]

bars = plt.bar(class_correlation.index, class_correlation.values, color=colors, edgecolor='black', linewidth=0.6)

# Tambahkan nilai angka presisi di atas/bawah setiap batang grafik
for bar in bars:
    yval = bar.get_height()
    va_dir = 'bottom' if yval >= 0 else 'top'
    y_offset = 0.02 if yval >= 0 else -0.05
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + y_offset, f'{yval:.2f}', 
             ha='center', va=va_dir, fontsize=8, fontweight='bold')

plt.axhline(0, color='black', linestyle='-', linewidth=0.8)
plt.title('Nilai Korelasi Pearson Seluruh Fitur Transaksi Terhadap Variabel Class (Fraud)', fontsize=14, pad=20, fontweight='bold')
plt.xlabel('Komponen Fitur (V1 - V28 & Amount)', fontsize=11)
plt.ylabel('Koefisien Korelasi', fontsize=11)
plt.ylim(-0.9, 0.9)
plt.xticks(rotation=45)

output_image = '/home/samid/korelasi_all_fitur_vs_class.png'
plt.savefig(output_image, dpi=300, bbox_inches='tight')
print(f"[SUKSES] Grafik Korelasi Total disimpan di: {output_image}")

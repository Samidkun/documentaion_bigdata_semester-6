import matplotlib.pyplot as plt
import seaborn as sns

# Data murni nilai korelasi absolut tertinggi terhadap Class dari log Skrip 1 & 2 lu
fitur = ['V4', 'V3', 'V1', 'V2', 'Amount']
nilai_korelasi = [0.7360, -0.6821, -0.5058, 0.4919, 0.0023]

sns.set_theme(style="whitegrid")
plt.figure(figsize=(9, 5))

# Plot bar horizontal agar mudah dibaca perbandingan fiturnya
colors = ['#e04a4a' if x > 0 else '#4a90e2' for x in nilai_korelasi]
bars = plt.barh(fitur, nilai_korelasi, color=colors, edgecolor='black', height=0.6)

# Tambahkan garis penanda nol
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

# Tambahkan nilai angka di ujung bar
for bar in bars:
    width = bar.get_width()
    x_pos = width + 0.02 if width >= 0 else width - 0.09
    plt.text(x_pos, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
             va='center', ha='center', fontsize=10, fontweight='bold')

plt.title('Top Fitur Berdasarkan Nilai Korelasi Terhadap Class', fontsize=13, pad=15, fontweight='bold')
plt.xlabel('Nilai Korelasi Pearson', fontsize=11)
plt.ylabel('Komponen Fitur', fontsize=11)
plt.xlim([-0.9, 0.9])

output_image = '/home/samid/top_features_korelasi.png'
plt.savefig(output_image, dpi=300, bbox_inches='tight')
print(f"[SUKSES] Grafik Top Fitur disimpan di: {output_image}")


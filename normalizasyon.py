import csv
import numpy as np

# Veri setini oku
with open('veri_seti.csv', 'r') as f:
    oku = csv.reader(f)
    veri = list(oku)

# Veri setindeki her sütun için normalizasyon yap
veri = np.array(veri).astype(float)
for i in range(veri.shape[1]-1):
    sutun = veri[:, i]
    veri[:, i] = (sutun - sutun.min()) / (sutun.max() - sutun.min())

# Normalize edilmiş veriyi yeni bir CSV dosyasına yaz
with open('normalize_veri.csv', 'w', newline='') as f:
    yaz = csv.writer(f)
    for satir in veri:
        # Her hücreyi 4 basamaklı bir sayı olarak formatla
        yeni_satir = ["%.4f" % x for x in satir]
        yaz.writerow(yeni_satir)

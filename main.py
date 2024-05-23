from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp

# CSV Dosyasını Yükle 
def csv_yukle(dosya):
	veri_seti = list()
	with open(dosya, 'r') as file:
		csv_oku = reader(file)
		for row in csv_oku:
			if not row:
				continue
			veri_seti.append(row)
	return veri_seti


# Veri Setindeki Değerleri Float Tipine Dönüştür
def stringi_floata_donustur(veri_seti, sutun):
	for satir in veri_seti:
		satir[sutun] = float(satir[sutun].strip())


# Veri Setindeki Etiket Değerlerini İnteger tipine Dönüştür
def stringi_inte_donustur(veri_seti, sutun):
	sinif_degeri = [satir[sutun] for satir in veri_seti]
	sinif = set(sinif_degeri)
	tamsayi = dict()
	for i, deger in enumerate(sinif):
		tamsayi[deger] = i
	for satir in veri_seti:
		satir[sutun] = tamsayi[satir[sutun]]
	return tamsayi


# Veri Setini Belirtilen Sayıya Böler ve 1 Tanesini Test Verisi Olarak Alır  
def veriyi_bolme(veri_seti, parca_sayisi):
	bolunmus_veri = list()
	veri_kopyasi = list(veri_seti)
	parca_boyutu = int(len(veri_seti) / parca_sayisi)
	for i in range(parca_sayisi):
		parca = list()
		while len(parca) < parca_boyutu:
			index = randrange(len(veri_kopyasi))
			parca.append(veri_kopyasi.pop(index))
		bolunmus_veri.append(parca)
	return bolunmus_veri


# Doğruluk Yüzdesini Hesapla
def dogruluk_yuzdesi(gercek, tahmin):
	dogruluk = 0
	for i in range(len(gercek)):
		if gercek[i] == tahmin[i]:
			dogruluk += 1
	return dogruluk / float(len(gercek)) * 100.0


# Test ve Eğitim Setini Oluşturur ve BAckPropagationdan gelen tahmin değerleri ile test verisinden gelen gerçek değerleri karşılaştırır. 
def temel_metod(veri, algoritma, parca_sayisi, *args):
  bolunmus_veri = veriyi_bolme(veri, parca_sayisi)
  basari_orani = list()
  egitim_verisi = list(bolunmus_veri)
  egitim_verisi.remove(bolunmus_veri[2])
  egitim_verisi = sum(egitim_verisi,[])
  test_verisi = list()
  for i in bolunmus_veri[2]:
    i_copy = list(i)
    test_verisi.append(i_copy)
    i_copy[-1] = 1
  
  tahmin = algoritma(egitim_verisi, test_verisi, *args)
  gercek = [i[-1] for i in bolunmus_veri[2]]
  basari = dogruluk_yuzdesi(gercek, tahmin)
  basari_orani.append(basari)

  return basari_orani


# e'nin Üzerindeki X'i Yani Modeli Hesaplar
def activate(agirlik, girdi):
	aktivasyon= agirlik[-1]
	for i in range(len(agirlik) - 1):
		aktivasyon += agirlik[i] * girdi[i]
	return aktivasyon


# Sigmoif Fonksiyonu
def sigmoid(aktivasyon):
	return 1.0 / (1.0 + exp(-aktivasyon))


# Verilen Girdiler ile Ağın İlk Çıktısını Bulur
def ileri_yayilim(ag, satir):
	girdi = satir
	for katman in ag:
		#bir önceki katmanın çıktıları bir sonraki katmanın girdileridir
		# bu yeni girdiler new_inputs dizisinde saklanır
		yeni_girdi = []
		for noron in katman:
			aktivasyon = activate(noron['weights'], girdi)
			noron['output'] = sigmoid(aktivasyon)
			yeni_girdi.append(noron['output'])
		girdi = yeni_girdi
	return girdi

# Bir Nöronun Çıktısının Türevi
def sigmoid_turev(cikti):
	return cikti * (1.0 - cikti)

# Düğümlerdeki hataları Hesaplar
def hatayi_yay(ag, gercek_deger):
	for i in reversed(range(len(ag))):
		katman = ag[i]
		hata_liste = list()
		if i != len(ag)-1:
			for j in range(len(katman)):
				hata = 0.0
				for noron in ag[i + 1]:
					hata += (noron['weights'][j] * noron['delta'])
				hata_liste.append(hata)

		else:
			for j in range(len(katman)):
				noron = katman[j]
				hata_liste.append(noron['output'] - gercek_deger[j])

		for j in range(len(katman)):
			noron = katman[j]
			noron['delta'] = hata_liste[j] * sigmoid_turev(noron['output'])


# Hataları Kullanarak Ağırlıkları Günceller
def agirliklari_guncelle(ag, satir, alfa):
	for i in range(len(ag)):
		girdi = satir[:-1]
		if i != 0:
			girdi = [noron['output'] for noron in ag[i - 1]]
		for noron in ag[i]:
			for j in range(len(girdi)):
				noron['weights'][j] -= alfa * noron['delta'] * girdi[j]
			noron['weights'][-1] -= alfa * noron['delta']

# Ağı Eğitir
def train_network(ag, train, alfa, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		for satir in train:
			outputs = ileri_yayilim(ag, satir)
			expected = [0 for i in range(n_outputs)]
			expected[satir[-1]] = 1
			hatayi_yay(ag, expected)
			agirliklari_guncelle(ag, satir, alfa)
		print(epoch,ag)
# Ağı Düğümleri ve Ağırlıkları Oluşturur
def initialize_network(girdi, g_noron_sayisi, cikti_sayisi):
	# n_hidden : gizli katmandaki neuron sayısı
	# n_inputs : girdi sayısı
	# n_outputs : çıktı sayısı
	# gizli katmandaki her nöron "n_inputs + 1" ağırlığa sahiptir. +1 : bias değeri
	# çıkış katmanındaki her nöron "n_hidden + 1" ağırlığa sahiptir
	ag = list()
	gizli_katman = [{'weights':[random() for i in range(girdi + 1)]} for i in range(g_noron_sayisi)]
	ag.append(gizli_katman)
	cikti_katmani = [{'weights':[random() for i in range(g_noron_sayisi + 1)]} for i in range(cikti_sayisi)]
	ag.append(cikti_katmani)
	return ag

# Tahmin Yapar
def tahmin_et(ag, satir):
	cikti = ileri_yayilim(ag, satir)
	return cikti.index(max(cikti))

# Gradient Descent Algoritması ile İşlemleri Gerçekleştirir
def back_propagation(train, test, alfa, epoch, g_noron_sayisi):
	girdi_sayisi = len(train[0]) - 1
	cikti_sayisi = len(set([row[-1] for row in train]))
	ag = initialize_network(girdi_sayisi, g_noron_sayisi, cikti_sayisi)
	train_network(ag, train, alfa, epoch, cikti_sayisi)
	tahmin_list = list()
	for row in test:
		tahmin = tahmin_et(ag, row)
		tahmin_list.append(tahmin)
	return(tahmin_list)

seed(1)
dosya = 'normalize_veri.csv'
veri_seti = csv_yukle(dosya)
for i in range(len(veri_seti[0])-1):
	stringi_floata_donustur(veri_seti, i)
stringi_inte_donustur(veri_seti, len(veri_seti[0])-1)
parca_sayisi = 4
alfa = 0.5  
epoch = 2200 
g_noron_sayisi = 2  
basari_orani = temel_metod(veri_seti, back_propagation, parca_sayisi, alfa, epoch, g_noron_sayisi)
print('Başarı Oranı: %s' % basari_orani)
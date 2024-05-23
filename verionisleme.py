import tensorflow as tf
import os
import numpy as np


dosya_yolu_tavuk = '/content/tavuk'
dosya_yolu_at    = '/content/at'
dosya_yolu_kedi  = '/content/kedi'


def tavuk_veri_seti(dosya_yolu_tavuk, verinin_sinifi):
  # Klasördeki resimler for dongüsü içerisnde okundu.
  for dosya in os.listdir(dosya_yolu_tavuk):
      # Resim okuma
      resim = tf.io.read_file(os.path.join(dosya_yolu_tavuk, dosya))
      # Resmi boyutlandır
      resim = tf.image.decode_jpeg(resim)
      resim = tf.image.resize(resim, (10, 10))
      # Renkli resmi renksiz hale getir
      resim = tf.image.rgb_to_grayscale(resim)
      # Resmi tek boyuta indir
      resim = tf.reshape(resim, (10*10,))
      resim = np.array(resim)
      #Veriyi Etiketle
      resim = np.append(resim,verinin_sinifi)
      # Resim verilerini csv dosyasına yaz
      csv_dosya = 'veri_seti.csv'
      with open(csv_dosya, 'a') as file:
        np.savetxt(file, resim.reshape(-1, resim.shape[-1]), delimiter=',')

tavuk_veri_seti(dosya_yolu_tavuk,0)


def at_veri_seti(dosya_yolu_at, verinin_sinifi):
  # Klasördeki resimleri oku
  for dosya in os.listdir(dosya_yolu_at):
      # Resmi oku
      resim = tf.io.read_file(os.path.join(dosya_yolu_at, dosya))
      # Resmi boyutlandır
      resim = tf.image.decode_jpeg(resim)
      resim = tf.image.resize(resim, (10, 10))
      # Renkli resmi renksiz hale getir
      resim = tf.image.rgb_to_grayscale(resim)
      # Resmi tek boyuta indir
      resim = tf.reshape(resim, (10*10,))
      resim = np.array(resim)
      #Veriyi Etiketle
      resim = np.append(resim,verinin_sinifi)
      # Resim verilerini csv dosyasına yaz
      csv_dosya = 'veri_seti.csv'
      with open(csv_dosya, 'a') as file:
        np.savetxt(file, resim.reshape(-1, resim.shape[-1]), delimiter=',')

at_veri_seti(dosya_yolu_at,1)


def kedi_veri_seti(dosya_yolu_kedi, verinin_sinifi):
  # Klasördeki resimleri oku
  for dosya in os.listdir(dosya_yolu_kedi):
      # Resmi oku
      resim = tf.io.read_file(os.path.join(dosya_yolu_kedi, dosya))
      # Resmi boyutlandır
      resim = tf.image.decode_jpeg(resim)
      resim = tf.image.resize(resim, (10, 10))
      # Renkli resmi renksiz hale getir
      resim = tf.image.rgb_to_grayscale(resim)
      # Resmi tek boyuta indir
      resim = tf.reshape(resim, (10*10,))
      resim = np.array(resim)
      #Veriyi Etiketle
      resim = np.append(resim,verinin_sinifi)
      # Resim verilerini csv dosyasına yaz
      csv_dosya = 'veri_seti.csv'
      with open(csv_dosya, 'a') as file:
        np.savetxt(file, resim.reshape(-1, resim.shape[-1]), delimiter=',') 

     
kedi_veri_seti(dosya_yolu_kedi,2)

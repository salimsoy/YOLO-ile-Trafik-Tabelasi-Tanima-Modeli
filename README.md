# 🚦 YOLO Tabanlı Trafik Tabelası Tanıma Modeli

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![YOLOv](https://img.shields.io/badge/Model-YOLOv8-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Otonom sürüş sistemleri ve ADAS uygulamaları için geliştirilmiş, **gerçek zamanlı trafik tabelası tanıma** derin öğrenme modeli.

---

## 🚀 Proje Genel Bakış

Bu proje, YOLO (You Only Look Once) mimarisini temel alarak trafik tabelalarını **yüksek doğruluk ve düşük gecikmeyle** tespit edip sınıflandırmak üzere tasarlanmıştır. Test ortamında **40+ FPS** performansıyla çalışarak otonom araçlar ve ADAS sistemleri için güvenilir bir karar destek mekanizması sunar.

---

## ✨ Temel Özellikler

- ⚡ **Gerçek Zamanlı Tespit** — YOLOvX tabanlı model ile anlık tespit ve sınıflandırma
- 🎯 **Yüksek Doğruluk** — Farklı aydınlatma ve çevresel koşullarda güvenilir performans
- 🔄 **Modüler Veri Pipeline'ı** — CSV etiketlerini otomatik olarak YOLO formatına dönüştürme
- 📷 **Webcam Entegrasyonu** — Canlı kamera akışı üzerinde gerçek zamanlı test imkânı
- 📓 **Jupyter Notebook Eğitimi** — İteratif model geliştirme ve performans iyileştirme ortamı

---

## 🧠 Teknik Mimari

### Model

YOLOvX mimarisi kullanılmaktadır. Model, trafik tabelalarını **tek bir ileri besleme geçişiyle** tespit ederek geleneksel iki aşamalı dedektörlere kıyasla üstün hız performansı sunar. Eğitim detayları `yolo_model.ipynb` içinde yer almaktadır.

### Veri Ön İşleme

| Betik | Açıklama |
|---|---|
| `txtceviri.py` | Ham CSV etiket verilerini YOLO `.txt` formatına dönüştürür; bounding box normalizasyonu ve sınıf ID atamasını içerir |
| `txttasi.py` | Etiket dosyalarını doğru dizin yapısına taşıyan yardımcı betik |

### Eğitim

Model, geniş ve çeşitli bir veri kümesi üzerinde eğitilmiştir. Genelleme yeteneğini artırmak için şu veri artırma teknikleri kullanılmıştır:

- Döndürme ve ölçekleme
- Parlaklık ve kontrast ayarı
- Yatay çevirme

Eğitilmiş model ağırlıkları: `run/detect/train/weights/best1.pt`

---

## 📊 Performans

| Metrik | Değer |
|---|---|
| İşleme Hızı | 40+ FPS (canlı kamera) |
| Ortam | Çeşitli aydınlatma koşulları |
| Uygulama Alanı | Otonom sürüş / ADAS |

---

## ⚙️ Kurulum

### Gereksinimler

- Python 3.x
- `ultralytics` (YOLOvX)
- `opencv-python`
- `pandas`

### Adımlar

```bash
# 1. Depoyu klonlayın
git clone https://github.com/salimsoy/YOLO-ile-Trafik-Tabelasi-Tanima-Modeli.git
cd YOLO-ile-Trafik-Tabelasi-Tanima-Modeli

# 2. Bağımlılıkları yükleyin
pip install ultralytics opencv-python pandas
# veya requirements.txt varsa:
pip install -r requirements.txt
```

---

## 🛠️ Kullanım

### Veri Ön İşleme (CSV → YOLO Formatı)

```bash
python txtceviri.py
```

### Model Eğitimi

```bash
jupyter notebook yolo_model.ipynb
```

### Gerçek Zamanlı Tespit (Webcam)

```bash
python yolo_test_webcam.py
```

---

## 📁 Dosya Yapısı

YOLO-ile-Trafik-Tabelasi-Tanima-Modeli/
├── run/
│   └── detect/
│       └── train/
│           └── weights/
│               └── best1.pt          # Eğitilmiş model ağırlıkları
├── dataset/                          # Veri setleri (örn. Train.csv)
├── txtceviri.py                      # CSV → YOLO format dönüştürücü
├── txttasi.py                        # Etiket dosyası taşıma betiği
├── yolo_model.ipynb                  # Model eğitim notebook'u
├── yolo_test_webcam.py               # Gerçek zamanlı webcam tespiti
└── README.md

---

## 🌍 Gerçek Dünya Etkisi

Bu model, otonom araçların ve ADAS sistemlerinin çevresel farkındalığını artırarak **sürüş güvenliğini iyileştirme** potansiyeline sahiptir. Yüksek FPS performansı ve düşük hata oranıyla yol üzerindeki kritik bilgilere anında tepki verilmesini sağlar.

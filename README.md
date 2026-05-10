# DeepGuard — Deepfake Yüz Tespit Sistemi

EfficientNet-B4 ve Xception + FFT + Cross-Attention mimarilerini birleştiren **feature-level fusion** modeli ile sahte yüz tespiti.

## Model Sonuçları

| Model | Genel Doğruluk | Gerçek | Sahte |
|-------|---------------|--------|-------|
| EfficientNet-B4 (FaceForensics++ pretrain) | %59.60 | %30.00 | %89.20 |
| EfficientNet-B4 (Celeb-DF fine-tune) | %96.80 | %94.40 | %99.20 |
| EfficientNet-B4 (Roop fine-tune) | %89.38 | %74.36 | %94.21 |
| **Fusion Modeli** | **%98.12** | **%94.87** | **%99.17** |

## Mimari

```
Giriş (299×299 RGB)
    │
    ├── EfficientNet-B4 → 1792-dim features
    │
    └── Xception + FFT + Cross-Attention
            ├── Xception       → 2048-dim
            ├── FFT Branch     →  512-dim
            └── Cross-Attention → 256-dim
                                = 2304-dim features
    │
    concat [1792 + 2304 = 4096-dim]
    │
    FC(512) → BN → ReLU → Dropout(0.4)
    │
    FC(1) → sigmoid → sahtelik skoru
```

## Veri Setleri

- **FaceForensics++**: Deepfakes, Face2Face, FaceSwap, NeuralTextures + gerçek YouTube videoları
- **Celeb-DF**: Ünlü deepfake videoları
- **Özel Roop veri seti**: LFW yüzleri üzerinde [Roop](https://github.com/s0md3v/roop) ile üretilmiş sahte + gerçek çiftler

## Kurulum

```bash
git clone https://github.com/<kullanici>/deepguard.git
cd deepguard

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
```

## Model Ağırlıkları

Model ağırlıkları repo'ya dahil edilmemiştir (boyut nedeniyle).  
Aşağıdaki dosyaları indirip repo kök dizinine yerleştirin:

| Dosya | Açıklama |
|-------|----------|
| `fusion_finetuned.pth` | Ana fusion modeli (EfficientNet-B4 + PixelGuard) |
| `efficientnet_finetuned.pth` | Standalone EfficientNet-B4 |
| `pixelguard_best.pth` | Standalone PixelGuard (Xception+FFT) |

## Demo Çalıştırma

```bash
python demo.py --weights fusion_finetuned.pth
# Tarayıcıda açılır: http://localhost:7860
```

Bir yüz fotoğrafı yükleyin → **Analiz Et** → GradCAM ısı haritası + sahtelik skoru.

## Proje Yapısı

```
deepguard/
├── fusion_model.py       # Fusion model mimarisi
├── demo.py               # Gradio web arayüzü
├── demo_compare.py       # 3 modeli karşılaştıran demo
│
├── train/                # Eğitim scriptleri
│   ├── train.py
│   ├── train_fusion.py
│   ├── fine_tune.py
│   ├── fine_tune_roop.py
│   ├── finetune_hard.py
│   └── finetune_real.py
│
├── eval/                 # Değerlendirme
│   ├── evaluate_all.py   # Accuracy / F1 / AUC-ROC karşılaştırması
│   ├── test_results.py
│   ├── test_folder.py
│   └── test_my_dataset.py
│
├── data/                 # Veri çekme scriptleri
│   ├── extractor.py
│   ├── celeb_extractor_pro.py
│   ├── roop_dataset_extractor.py
│   └── ...
│
├── visualize/            # Görselleştirme
│   ├── visualize.py      # GradCAM ısı haritası üretimi
│   ├── visualize_results.py
│   ├── prepare_demo_folder.py
│   └── make_report.py
│
├── roop_scripts/         # Roop entegrasyon scriptleri
│   ├── roop_batch.py
│   ├── roop_taha.py
│   └── roop_taha_fast.py
│
└── gradcam_output/       # Örnek GradCAM görselleştirmeleri
```

## GradCAM Örnekleri

`gradcam_output/` klasöründe modelin gerçek ve sahte yüzlerde odaklandığı bölgeler görselleştirilmiştir.

## Lisans

Bu proje akademik amaçlıdır.

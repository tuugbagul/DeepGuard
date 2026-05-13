# DeepGuard

DeepGuard, bir yüz fotoğrafının deepfake olup olmadığını tahmin eden bir Gradio demosudur. Model tarafında `EfficientNet-B4` ile `Xception + FFT + Cross-Attention` birleşiminden oluşan bir fusion mimarisi kullanır.

Bu repo artık öncelikle **demo/inference** deneyimine odaklıdır. Eğitim, veri hazırlama ve değerlendirme scriptleri hâlâ repoda durur; ancak onlar ileri seviye/araştırma kullanımı içindir.

## Hızlı Başlangıç

```bash
git clone https://github.com/tuugbagul/DeepGuard.git
cd DeepGuard

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
python demo.py
```

Windows'ta isterseniz doğrudan şunu da çalıştırabilirsiniz:

```bat
run_demo.bat
```

Demo açıldığında tarayıcıda `http://127.0.0.1:7860` adresine gider. Bir yüz fotoğrafı yükleyin, model tahmini ve GradCAM ısı haritasını görün.

## Weight Akışı

Demo, `fusion_finetuned.pth` dosyasını şu sırayla arar:

1. `--weights` ile verdiğiniz özel yol
2. Repo kökündeki `fusion_finetuned.pth`
3. `weights/fusion_finetuned.pth`
4. GitHub Release asset indirimi

Varsayılan release URL şablonu:

```text
https://github.com/tuugbagul/DeepGuard/releases/latest/download/fusion_finetuned.pth
```

Eğer release asset henüz yayınlanmadıysa iki kolay seçenek var:

```bash
python demo.py --weights path/to/fusion_finetuned.pth
```

veya

```bash
set DEEPGUARD_WEIGHTS_URL=https://your-direct-file-url/fusion_finetuned.pth
python demo.py
```

PowerShell için:

```powershell
$env:DEEPGUARD_WEIGHTS_URL="https://your-direct-file-url/fusion_finetuned.pth"
python demo.py
```

## Sık Kullanılan Komutlar

```bash
python demo.py
python demo.py --no-download
python demo.py --weights weights/fusion_finetuned.pth
python demo.py --host 0.0.0.0 --port 7860
```

## Model Sonuçları

| Model | Genel Doğruluk | Gerçek | Sahte |
|-------|---------------|--------|-------|
| EfficientNet-B4 (FaceForensics++ pretrain) | %59.60 | %30.00 | %89.20 |
| EfficientNet-B4 (Celeb-DF fine-tune) | %96.80 | %94.40 | %99.20 |
| EfficientNet-B4 (Roop fine-tune) | %89.38 | %74.36 | %94.21 |
| **Fusion Modeli** | **%98.12** | **%94.87** | **%99.17** |

## Mimari

```text
Giriş (299x299 RGB)
    |
    |-- EfficientNet-B4 -> 1792-dim features
    |
    `-- Xception + FFT + Cross-Attention
            |-- Xception        -> 2048-dim
            |-- FFT Branch      ->  512-dim
            `-- Cross-Attention ->  256-dim
                                   = 2304-dim features

concat [1792 + 2304 = 4096-dim]
    -> FC(512) -> BN -> ReLU -> Dropout(0.4)
    -> FC(1) -> sigmoid -> sahtelik skoru
```

## Veri Setleri

- **FaceForensics++**: Deepfakes, Face2Face, FaceSwap, NeuralTextures ve gerçek YouTube videoları
- **Celeb-DF**: Ünlü deepfake videoları
- **Özel Roop veri seti**: LFW yüzleri üzerinde [Roop](https://github.com/s0md3v/roop) ile üretilmiş sahte + gerçek çiftler

## Repo Yapısı

```text
DeepGuard/
├── demo.py               # Ana Gradio demo girişi
├── weights.py            # Weight çözümleme / indirme yardımcıları
├── fusion_model.py       # Fusion model mimarisi
├── pixelguard_model.py   # PixelGuard kolu
├── train/                # Eğitim scriptleri
├── eval/                 # Değerlendirme scriptleri
├── data/                 # Veri hazırlama scriptleri
├── visualize/            # GradCAM ve raporlama araçları
└── roop_scripts/         # Roop üretim / yardımcı scriptleri
```

## Araştırma Scriptleri

Bu klasörler demo akışının parçası değildir:

- `train/`
- `eval/`
- `data/`
- `visualize/`
- `roop_scripts/`

Onlar model geliştirme ve deney çalışmaları için tutulur. İlk kurulumda yalnızca `demo.py` ile ilgilenmeniz yeterlidir.

## Lisans

Bu proje akademik amaçlıdır.

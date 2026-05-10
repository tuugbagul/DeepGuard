"""
Üç modeli aynı test setinde karşılaştırır:
  1. EfficientNet-B4 (roop fine-tune)
  2. PixelGuard (Xception + FFT + CrossAttention)
  3. Fusion (EfficientNet + PixelGuard)

Test seti: celeb_pro_6000.json — fusion modeli bunu hiç görmedi.
"""
import sys, json, random, cv2, os
import numpy as np
import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

from fusion_model import FusionModel
from pixelguard_model import PixelGuardHybridModel

# ── YOL YAPILANDIRMASI ───────────────────────────────────────────────────────
# Bu yolları kendi ortamınıza göre düzenleyin
CELEB_JSON     = 'celeb_pro_6000.json'
EFF_WEIGHTS    = 'efficientnet_finetuned.pth'
PG_WEIGHTS     = 'pixelguard_best.pth'
FUSION_WEIGHTS = 'fusion_best.pth'
# ─────────────────────────────────────────────────────────────────────────────

TEST_SAMPLES = 500   # celeb'den rastgele 500 kare al (hız için)
BATCH_SIZE   = 16


# ---------------------------------------------------------------
class GenericDataset(Dataset):
    """
    mode='eff'    : 224x224, ImageNet normalize
    mode='pg'     : 299x299, [-1,1] normalize
    mode='fusion' : 299x299, [0,1] ham (model içinde normalize eder)
    """
    def __init__(self, samples, mode='fusion'):
        self.samples = samples
        self.mode = mode
        if mode == 'eff':
            self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            self.size = 224
        elif mode == 'pg':
            self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            self.std  = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            self.size = 299
        else:  # fusion
            self.size = 299

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            with Image.open(s['img']) as pil:
                img = np.array(pil.convert('RGB'))
            img = cv2.resize(img, (self.size, self.size))
        except Exception:
            img = np.zeros((self.size, self.size, 3), dtype=np.uint8)

        img = img.astype(np.float32) / 255.0
        if self.mode in ('eff', 'pg'):
            img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img), torch.tensor(s['label'], dtype=torch.float32)


# ---------------------------------------------------------------
class EffDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=1)
    def forward(self, x):
        return self.model(x).squeeze(1)


def run_test(model, loader, device, desc):
    model.eval()
    correct, total = 0, 0
    real_c, real_t, fake_c, fake_t = 0, 0, 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=desc, ncols=70):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            if logits.dim() > 1:
                logits = logits.squeeze(1)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            for p, l in zip(preds, labels):
                if l == 1:
                    fake_t += 1
                    if p == l: fake_c += 1
                else:
                    real_t += 1
                    if p == l: real_c += 1
    return {
        'genel':  100 * correct / total,
        'gercek': 100 * real_c / real_t if real_t else 0,
        'sahte':  100 * fake_c / fake_t if fake_t else 0,
        'total':  total,
    }


# ---------------------------------------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cihaz: {device}")

    # Test verisini hazırla
    with open(CELEB_JSON) as f:
        data = json.load(f)
    random.seed(42)
    random.shuffle(data)
    test_data = data[:TEST_SAMPLES]
    real_n = sum(1 for s in test_data if s['label'] == 0)
    fake_n = sum(1 for s in test_data if s['label'] == 1)
    print(f"\nTest seti: {TEST_SAMPLES} ornek — Gercek: {real_n}, Sahte: {fake_n}")
    print("Kaynak: celeb_pro_6000.json (fusion modeli hic gormedi)\n")

    results = {}

    # ---- 1. EfficientNet ----
    eff_model = EffDetector().to(device)
    eff_model.load_state_dict(torch.load(EFF_WEIGHTS, map_location=device))
    eff_loader = DataLoader(GenericDataset(test_data, mode='eff'),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    results['EfficientNet-B4'] = run_test(eff_model, eff_loader, device, 'EfficientNet')
    del eff_model

    # ---- 2. PixelGuard ----
    pg_model = PixelGuardHybridModel(pretrained=False).to(device)
    pg_model.load_state_dict(torch.load(PG_WEIGHTS, map_location=device))
    pg_loader = DataLoader(GenericDataset(test_data, mode='pg'),
                           batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    results['PixelGuard'] = run_test(pg_model, pg_loader, device, 'PixelGuard  ')
    del pg_model

    # ---- 3. Fusion ----
    fusion_model = FusionModel().to(device)
    fusion_model.load_state_dict(torch.load(FUSION_WEIGHTS, map_location=device))
    f_loader = DataLoader(GenericDataset(test_data, mode='fusion'),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    results['Fusion'] = run_test(fusion_model, f_loader, device, 'Fusion      ')
    del fusion_model

    # ---- Rapor ----
    print("\n" + "=" * 55)
    print("  MODEL KARSILASTIRMA — Celeb-DF Test Seti")
    print("=" * 55)
    print(f"  {'Model':<20} {'Genel':>7}  {'Gercek':>7}  {'Sahte':>7}")
    print("-" * 55)
    for name, r in results.items():
        print(f"  {name:<20} %{r['genel']:>6.2f}  %{r['gercek']:>6.2f}  %{r['sahte']:>6.2f}")
    print("=" * 55)
    print(f"  Test ornegi: {TEST_SAMPLES}  |  Gercek: {real_n}  |  Sahte: {fake_n}")
    print("=" * 55)


if __name__ == '__main__':
    main()

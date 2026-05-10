import os, json, cv2
import torch
import torch.nn as nn
import timm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image


# ==========================================
# 1. VERİ SETİ SINIFI
# ==========================================
class FFPPDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # 1. Klasör yolunu ve dosya adını al
        img_name = os.path.basename(s['img'])
        FRAMES_DIR = os.environ.get('CELEB_FRAMES_DIR', 'Celeb-DF_Test_Frames')
        full_path = os.path.join(FRAMES_DIR, img_name)

        # 2. OpenCV yerine PIL (Pillow) kullanarak oku
        try:
            # Pillow dosya yolundaki karakterleri OpenCV'den daha iyi yönetir
            with Image.open(full_path) as pil_img:
                img = np.array(pil_img.convert('RGB'))  # RGB formatına çevir ve numpy yap

            # 3. Yeniden boyutlandır (PIL sonrası numpy olduğu için cv2 kullanabiliriz)
            img = cv2.resize(img, (224, 224))

        except Exception as e:
            # Eğer burası da hata verirse, sorun dosyanın kendisinde veya Windows iznindedir
            print(f"PIL Okuma Hatası: {img_name} -> {e}")
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        # 4. Normalizasyon
        img = ((img.astype(np.float32) / 255.0 - self.mean) / self.std).transpose(2, 0, 1)
        return torch.from_numpy(img), torch.tensor(s['label'], dtype=torch.float32)
# ==========================================
# 2. MODEL MİMARİSİ (Eğitim kodunla birebir aynı!)
# ==========================================
class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        # Eğitimde 'self.model' ismini kullandığın için burada da aynısını kullanıyoruz
        self.model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=1)

    def forward(self, x):
        return self.model(x).squeeze(1)


# ==========================================
# 3. TEST FONKSİYONU
# ==========================================
def final_test():
    # GPU Ayarı
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cihaz: {device}")

    # Modeli Başlat
    model = Detector().to(device)

    # Dosya Yolu (Eğitim kodundaki isme göre güncellendi)
    model_path = 'checkpoint_ep30.pth'  # kendi ağırlık dosyanıza göre düzenleyin

    if not os.path.exists(model_path):
        print(f"HATA: Model dosyası bulunamadı: {model_path}")
        return

    # Ağırlıkları Yükle
    try:
        model.load_state_dict(torch.load(model_path))
        print("Model başarıyla yüklendi!")
    except Exception as e:
        print(f"Yükleme hatası: {e}")
        return

    model.eval()

    # JSON dosyasını yükle (Yolun doğru olduğundan emin ol)
    json_path = 'celeb_test.json'
    if not os.path.exists(json_path):
        print(f"HATA: {json_path} bulunamadı!")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # DataLoader
    loader = DataLoader(FFPPDataset(data), batch_size=32, shuffle=False, num_workers=0)

    correct = 0
    total = 0
    fake_correct, fake_total = 0, 0
    real_correct, real_total = 0, 0

    print("Celeb-DF Testi Başlıyor...")
    with torch.no_grad():
        for imgs, labels in tqdm(loader):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            preds = (torch.sigmoid(outputs) > 0.85).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for p, l in zip(preds, labels):
                if l == 1:
                    fake_total += 1
                    if p == l: fake_correct += 1
                else:
                    real_total += 1
                    if p == l: real_correct += 1

    # SONUÇLARI YAZDIR
    print("\n" + "=" * 30)
    print(f"TOPLAM TEST EDİLEN KARE: {total}")
    print(f"GENEL BAŞARI: %{100 * correct / total:.2f}")
    if real_total > 0:
        print(f"GERÇEK VİDEO BAŞARISI: %{100 * real_correct / real_total:.2f}")
    if fake_total > 0:
        print(f"SAHTE VİDEO BAŞARISI: %{100 * fake_correct / fake_total:.2f}")
    print("=" * 30)


if __name__ == "__main__":
    final_test()
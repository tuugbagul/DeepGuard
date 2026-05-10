import os, json, cv2
import torch
import torch.nn as nn
import timm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            with Image.open(s['img']) as pil_img:
                img = np.array(pil_img.convert('RGB'))
            img = cv2.resize(img, (224, 224))
        except Exception as e:
            print(f"Okuma hatası: {s['img']} -> {e}")
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        img = ((img.astype(np.float32) / 255.0 - self.mean) / self.std).transpose(2, 0, 1)
        return torch.from_numpy(img), torch.tensor(s['label'], dtype=torch.float32)


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=1)

    def forward(self, x):
        return self.model(x).squeeze(1)


def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cihaz: {device}")

    model = Detector().to(device)
    model_path = 'efficientnet_finetuned.pth'  # kendi ağırlık dosyanıza göre düzenleyin

    if not os.path.exists(model_path):
        print(f"HATA: Model bulunamadı: {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with open('roop_dataset.json', 'r') as f:
        data = json.load(f)

    loader = DataLoader(MyDataset(data), batch_size=8, shuffle=False, num_workers=0)

    correct, total = 0, 0
    fake_correct, fake_total = 0, 0
    real_correct, real_total = 0, 0

    print("\nTest başlıyor...")
    with torch.no_grad():
        for imgs, labels in tqdm(loader):
            imgs, labels = imgs.to(device), labels.to(device)
            preds = (torch.sigmoid(model(imgs)) > 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for p, l in zip(preds, labels):
                if l == 1:
                    fake_total += 1
                    if p == l: fake_correct += 1
                else:
                    real_total += 1
                    if p == l: real_correct += 1

    print("\n" + "=" * 35)
    print(f"TOPLAM: {total}")
    print(f"GENEL BAŞARI:  %{100 * correct / total:.2f}")
    if real_total > 0:
        print(f"GERÇEK BAŞARI: %{100 * real_correct / real_total:.2f}  ({real_correct}/{real_total})")
    if fake_total > 0:
        print(f"SAHTE BAŞARI:  %{100 * fake_correct / fake_total:.2f}  ({fake_correct}/{fake_total})")
    print("=" * 35)


if __name__ == "__main__":
    test()

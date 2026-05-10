# Test setinden ornek gorseller uretir:
# sol: orijinal, sag: GradCAM isi haritasi
# alt bant: GERCEK / TAHMIN / skor / OK-FAIL
# python visualize_results.py

import os, sys, json, random
import numpy as np
import torch
import cv2
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from fusion_model import FusionModel

# ── YOL YAPILANDIRMASI ───────────────────────────────────────────────────────
# Bu yolları kendi ortamınıza göre düzenleyin
FUSION_WEIGHTS = 'fusion_finetuned.pth'
ROOP_JSON      = 'roop_dataset.json'
CELEB_JSON     = 'celeb_pro_6000.json'
OUT_DIR        = 'demo_gradcam_output'
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLD      = 0.6
N_EACH         = 5   # TP, TN, FN, FP icin kac tane

os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FusionModel().to(device)
model.load_state_dict(torch.load(FUSION_WEIGHTS, map_location=device))
model.eval()
print(f"Model hazir ({device})")


class GradCAM:
    def __init__(self, m):
        self.model = m
        self.activations = None
        self.gradients = None
        target = m.efficientnet.conv_head
        target.register_forward_hook(self._fwd)
        target.register_full_backward_hook(self._bwd)

    def _fwd(self, module, inp, out):
        self.activations = out

    def _bwd(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def run(self, tensor):
        self.model.zero_grad()
        logit = self.model(tensor).squeeze()
        score = torch.sigmoid(logit).item()
        logit.backward()
        w = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.activations).sum(dim=1).squeeze())
        cam = cam.cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, score


gradcam = GradCAM(model)

DISPLAY_SIZE = 300   # her panetin genisligi ve yuksekligi
BAR_H        = 36   # alt bilgi bangi yuksekligi
GREEN  = (0, 220, 80)
RED    = (60, 60, 255)
WHITE  = (255, 255, 255)
FONT   = cv2.FONT_HERSHEY_SIMPLEX


def make_card(img_path, label):
    try:
        with Image.open(img_path) as pil:
            img_rgb = np.array(pil.convert('RGB'))
    except Exception:
        return None, None

    img_299 = cv2.resize(img_rgb, (299, 299))
    tensor  = torch.from_numpy(img_299.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    cam, score = gradcam.run(tensor)

    # Orijinal gorsel (DISPLAY_SIZE x DISPLAY_SIZE)
    orig = cv2.resize(img_rgb, (DISPLAY_SIZE, DISPLAY_SIZE))

    # Heatmap
    cam_up   = cv2.resize(cam, (DISPLAY_SIZE, DISPLAY_SIZE))
    heatmap  = cv2.applyColorMap((cam_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended  = (0.55 * orig + 0.45 * heat_rgb).clip(0, 255).astype(np.uint8)

    # Yan yana birlestir
    side_by_side = np.concatenate([orig, blended], axis=1)  # (300, 600, 3)

    # Alt bilgi banti
    pred   = 1 if score > THRESHOLD else 0
    ok     = pred == label
    bar    = np.zeros((BAR_H, DISPLAY_SIZE * 2, 3), dtype=np.uint8)

    true_str = "SAHTE" if label == 1 else "GERCEK"
    pred_str = "SAHTE" if pred  == 1 else "GERCEK"
    text = f"GERCEK: {true_str}  |  TAHMIN: {pred_str} (skor:{score:.2f})  |  {'OK' if ok else 'FAIL'}"
    color = GREEN if ok else RED
    cv2.putText(bar, text, (8, 24), FONT, 0.52, color, 1, cv2.LINE_AA)

    card = np.concatenate([side_by_side, bar], axis=0)
    return card, (score, pred, ok)


def build_test_set():
    with open(ROOP_JSON) as f: roop = json.load(f)
    with open(CELEB_JSON) as f: celeb = json.load(f)
    random.seed(42)
    random.shuffle(roop)
    roop_test  = roop[int(len(roop) * 0.8):]
    celeb_test = celeb[1500:]
    samples = [s for s in roop_test + celeb_test if os.path.exists(s['img'])]
    print(f"Test seti: {len(samples)} ornek")
    return samples


samples = build_test_set()
random.seed(7)
random.shuffle(samples)

tp, tn, fn, fp = [], [], [], []

print(f"Gorsel uretiliyor (ilk 1000 ornek taranacak)...")
for s in samples[:1000]:
    if all(len(x) >= N_EACH for x in [tp, tn, fn, fp]):
        break
    card, info = make_card(s['img'], s['label'])
    if card is None:
        continue
    score, pred, ok = info
    label = s['label']
    entry = (card, s['img'], score)
    if   label == 1 and pred == 1 and len(tp) < N_EACH: tp.append(entry)
    elif label == 0 and pred == 0 and len(tn) < N_EACH: tn.append(entry)
    elif label == 1 and pred == 0 and len(fn) < N_EACH: fn.append(entry)
    elif label == 0 and pred == 1 and len(fp) < N_EACH: fp.append(entry)

print(f"  TP:{len(tp)}  TN:{len(tn)}  FN:{len(fn)}  FP:{len(fp)}")


def save_group(lst, prefix):
    for i, (card, path, score) in enumerate(lst):
        base = os.path.splitext(os.path.basename(path))[0]
        dst  = os.path.join(OUT_DIR, f"{prefix}__{i+1}__{base}__skor{int(score*100)}.jpg")
        cv2.imwrite(dst, cv2.cvtColor(card, cv2.COLOR_RGB2BGR))


save_group(tp, "1_DOGRU_SAHTE")
save_group(tn, "2_DOGRU_GERCEK")
save_group(fn, "3_YANLIS_KACIRILAN_SAHTE")
save_group(fp, "4_YANLIS_ALARM_GERCEK")

total = len(tp) + len(tn) + len(fn) + len(fp)
print(f"\nKaydedildi: {total} kart  ->  {OUT_DIR}")
for f in sorted(os.listdir(OUT_DIR)):
    print(" ", f)

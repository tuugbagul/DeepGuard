"""
GradCAM ile fusion modelinin hangi bölgeye baktığını görselleştirir.
Kullanım:
  python visualize.py                        # roop_dataset'ten 4 örnek (2 gerçek, 2 sahte)
  python visualize.py "C:/yol/resim.jpg"     # tek resim
"""
import sys, os, json, random, cv2
import numpy as np
import torch
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from fusion_model import FusionModel

# ── YOL YAPILANDIRMASI ───────────────────────────────────────────────────────
# Bu yolları kendi ortamınıza göre düzenleyin
FUSION_WEIGHTS = 'fusion_finetuned.pth'
ROOP_JSON      = 'roop_dataset.json'
OUTPUT_DIR     = 'gradcam_output'
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------
class GradCAM:
    """EfficientNet backbone'unun son conv katmanından ısı haritası üretir."""

    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None

        # EfficientNet'in son 1×1 conv katmanı (1792 kanal, en iyi spatial kapsam)
        target = model.efficientnet.conv_head
        target.register_forward_hook(self._fwd_hook)
        target.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.activations = out  # (1, C, H, W)

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]  # (1, C, H, W)

    def generate(self, img_tensor):
        """
        img_tensor: (1, 3, 299, 299), [0,1], device'da
        Döner: cam (H, W) numpy, 0-1 normalize edilmiş
        """
        self.model.zero_grad()
        logit = self.model(img_tensor).squeeze()
        logit.backward()  # sahte olma skoruna göre gradyan

        # Kanal ağırlıkları: gradyanların uzaysal ortalaması
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1).squeeze()   # (H, W)
        cam = torch.relu(cam).cpu().detach().numpy()

        # 0-1 normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ---------------------------------------------------------------
def load_image(path):
    """Resmi 299×299 tensöre çevirir, [0,1]."""
    with Image.open(path) as pil:
        img_rgb = np.array(pil.convert('RGB'))
    img_299 = cv2.resize(img_rgb, (299, 299))
    tensor = torch.from_numpy(
        img_299.astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0)
    return img_rgb, tensor  # orijinal boyut + model tensörü


def overlay_heatmap(original_rgb, cam):
    """CAM'ı orijinal görüntünün üstüne bindirir."""
    h, w = original_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = (0.55 * original_rgb + 0.45 * heatmap_rgb).clip(0, 255).astype(np.uint8)
    return blended


def make_panel(original_rgb, blended, label_true, label_pred, score):
    """Yan yana orijinal | GradCAM paneli üretir, etiket ekler."""
    h = 320
    orig_r = cv2.resize(original_rgb, (h, h))
    blen_r = cv2.resize(blended, (h, h))

    panel = np.concatenate([orig_r, blen_r], axis=1)  # (320, 640, 3)

    pred_str  = "GERCEK" if label_pred == 0 else "SAHTE"

    if label_true == -1:
        truth_str = "?"
        correct   = ""
        color     = (180, 180, 180)
    else:
        truth_str = "GERCEK" if label_true == 0 else "SAHTE"
        correct   = "OK" if label_true == label_pred else "YANLIS"
        color     = (60, 200, 60) if label_true == label_pred else (220, 60, 60)

    # Siyah şerit alt bilgi
    footer = np.zeros((50, panel.shape[1], 3), dtype=np.uint8)
    suffix = f"  |  {correct}" if correct else ""
    text = f"GERCEK: {truth_str}  |  TAHMIN: {pred_str} (skor:{score:.2f}){suffix}"
    cv2.putText(footer, text, (10, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    return np.concatenate([panel, footer], axis=0)


# ---------------------------------------------------------------
def run(image_paths_labels):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FusionModel().to(device)
    model.load_state_dict(torch.load(FUSION_WEIGHTS, map_location=device))
    model.eval()

    gradcam = GradCAM(model)

    saved = []
    for path, true_label in image_paths_labels:
        original_rgb, tensor = load_image(path)
        tensor = tensor.to(device)

        cam = gradcam.generate(tensor)
        score = torch.sigmoid(model(tensor)).item()
        pred_label = 1 if score > 0.5 else 0

        blended = overlay_heatmap(original_rgb, cam)
        panel   = make_panel(original_rgb, blended, true_label, pred_label, score)

        fname = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"gradcam_{fname}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        saved.append(out_path)

        truth_str = "?" if true_label == -1 else ("GERCEK" if true_label == 0 else "SAHTE")
        pred_str  = "GERCEK" if pred_label == 0 else "SAHTE"
        print(f"[{truth_str}] tahmin={pred_str} skor={score:.3f}  ->  {out_path}")

    return saved


# ---------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Tek resim modu: visualize.py resim.jpg [0|1]
        img_path = sys.argv[1]
        label = int(sys.argv[2]) if len(sys.argv) > 2 else -1
        samples = [(img_path, label)]
    else:
        # roop_dataset'ten 2 gerçek + 2 sahte seç
        with open(ROOP_JSON) as f:
            data = json.load(f)
        random.seed(7)
        real_s = random.sample([s for s in data if s['label'] == 0], 2)
        fake_s = random.sample([s for s in data if s['label'] == 1], 2)
        samples = [(s['img'], s['label']) for s in real_s + fake_s]

    print(f"GradCAM isleniyor: {len(samples)} goruntu")
    paths = run(samples)
    print(f"\nKaydedilen klasor: {OUTPUT_DIR}")
    print("Dosyalar:", *[os.path.basename(p) for p in paths], sep='\n  ')

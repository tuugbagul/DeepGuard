"""
Deepfake Tespit Demo
Çalıştır: python demo.py
Tarayıcıda açılır: http://localhost:7860
"""
import os, sys, cv2, argparse
import numpy as np
import torch
import gradio as gr
from PIL import Image

from fusion_model import FusionModel

parser = argparse.ArgumentParser(description='DeepGuard Deepfake Tespit Demo')
parser.add_argument('--weights', default='fusion_finetuned.pth',
                    help='Fusion model ağırlık dosyası (.pth)')
args = parser.parse_args()
FUSION_WEIGHTS = args.weights

print("Model yükleniyor...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FusionModel().to(device)
model.load_state_dict(torch.load(FUSION_WEIGHTS, map_location=device))
model.eval()
print(f"Model hazır. Cihaz: {device}")


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        target = model.efficientnet.conv_head
        target.register_forward_hook(self._fwd_hook)
        target.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.activations = out

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, img_tensor):
        self.model.zero_grad()
        logit = self.model(img_tensor).squeeze()
        score = torch.sigmoid(logit).item()
        logit.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = torch.relu(cam).cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, score


gradcam = GradCAM(model)


def predict(pil_image):
    if pil_image is None:
        return None, "<div style='text-align:center;padding:20px;font-size:18px;color:gray;'>Fotoğraf yükleyin</div>"

    img_rgb = np.array(pil_image.convert('RGB'))
    img_299 = cv2.resize(img_rgb, (299, 299))
    tensor = torch.from_numpy(
        img_299.astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(device)

    cam, score = gradcam.generate(tensor)

    h, w = img_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = (0.55 * img_rgb + 0.45 * heatmap_rgb).clip(0, 255).astype(np.uint8)

    pred = "SAHTE" if score > 0.6 else "GERÇEK"
    confidence = score if pred == "SAHTE" else 1.0 - score
    color = "#e53e3e" if pred == "SAHTE" else "#38a169"
    bar_color = "#e53e3e" if pred == "SAHTE" else "#38a169"
    bar_pct = int(score * 100)

    result_html = f"""
    <div style="font-family:sans-serif; padding:16px;">
        <div style="text-align:center; padding:18px; border-radius:12px;
             background:{color}; color:white; font-size:32px; font-weight:bold;
             letter-spacing:2px; margin-bottom:16px;">
            {pred}
        </div>
        <div style="margin-bottom:8px; font-size:14px; color:#555;">
            Sahtelik Skoru
        </div>
        <div style="background:#eee; border-radius:8px; height:22px; overflow:hidden; margin-bottom:6px;">
            <div style="width:{bar_pct}%; background:{bar_color}; height:100%; border-radius:8px;
                 transition:width 0.3s;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:13px; color:#777;">
            <span>0% — GERÇEK</span>
            <span><b>{score*100:.1f}%</b></span>
            <span>100% — SAHTE</span>
        </div>
        <div style="margin-top:14px; font-size:13px; color:#888; text-align:center;">
            Güven: <b>%{confidence*100:.1f}</b>
        </div>
    </div>
    """

    return Image.fromarray(blended), result_html


with gr.Blocks(title="Deepfake Tespit Sistemi", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔍 Deepfake Yüz Tespit Sistemi
    **EfficientNet-B4 + PixelGuard** füzyon modeli ile fotoğrafın sahte mi gerçek mi olduğunu tespit eder.
    GradCAM ısı haritası modelin hangi bölgeye odaklandığını gösterir.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Fotoğraf Yükle", height=360)
            btn = gr.Button("Analiz Et", variant="primary", size="lg")

        with gr.Column(scale=1):
            img_output = gr.Image(type="pil", label="GradCAM Isı Haritası", height=360)
            result_html = gr.HTML()

    btn.click(fn=predict, inputs=img_input, outputs=[img_output, result_html])
    img_input.change(fn=predict, inputs=img_input, outputs=[img_output, result_html])

    gr.Markdown("""
    ---
    **Nasıl kullanılır:** Bir yüz fotoğrafı yükleyin → *Analiz Et* butonuna tıklayın.
    Sağda GradCAM ısı haritası ve sahtelik skoru görünür.
    """)

if __name__ == '__main__':
    demo.launch(inbrowser=True)

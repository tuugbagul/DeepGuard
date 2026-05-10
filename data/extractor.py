import cv2
import os
import json
import random
from pathlib import Path
from tqdm import tqdm
import numpy as np

# --- YOLLAR ---
# Drive Desktop yolu (G: sürücüsü)
FFPP_ROOT = r'G:\My Drive\pixelguard\data\faceforensics'

# Çıktı yolu (Burası yerel diskin olsun, örn: Masaüstü veya D sürücüsü)
# Drive içine yazmak yavaşlatabilir, o yüzden önce yerele almanı öneririm.
DATA_ROOT = r'C:\pixelguard_frames'
LIST_DIR = os.path.join(DATA_ROOT, 'lists')

os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(LIST_DIR, exist_ok=True)

# Ayarlar
FRAMES_PER_VIDEO = 30
IMG_SIZE = 224
FAKE_TYPES = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']


def extract_frames(video_path, out_dir, n_frames=30, size=224):
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) >= n_frames:
        return [os.path.join(out_dir, f) for f in os.listdir(out_dir)]
    
    """Videodan kareleri çeker ve kaydeder."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Eğer klasörde zaten yeterli kare varsa işlemi atla (Hız kazandırır)
    if len(os.listdir(out_dir)) >= n_frames:
        return [os.path.join(out_dir, f) for f in os.listdir(out_dir)]

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = set([int(i * total / n_frames) for i in range(n_frames)])
    saved = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx in indices:
            frame = cv2.resize(frame, (size, size))
            out_path = os.path.join(out_dir, f'frame_{frame_idx:05d}.jpg')
            cv2.imwrite(out_path, frame)
            saved.append(out_path)
        frame_idx += 1

    cap.release()
    return saved


samples = []

# --- İŞLEME DÖNGÜSÜ ---
# 1. Fake Videolar
for fake_type in FAKE_TYPES:
    video_dir = os.path.join(FFPP_ROOT, 'manipulated_sequences', fake_type, 'c40', 'videos')
    if not os.path.exists(video_dir):
        print(f"Uyarı: {video_dir} bulunamadı.")
        continue

    videos = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
    for vid_file in tqdm(videos, desc=f"İşleniyor: {fake_type}"):
        vid_path = os.path.join(video_dir, vid_file)
        stem = Path(vid_file).stem
        target_dir = os.path.join(DATA_ROOT, 'frames', fake_type, stem)

        frame_paths = extract_frames(vid_path, target_dir, FRAMES_PER_VIDEO, IMG_SIZE)
        for fp in frame_paths:
            samples.append({'img': fp, 'label': 1, 'type': fake_type})

# 2. Real Videolar
real_dir = os.path.join(FFPP_ROOT, 'original_sequences', 'youtube', 'c40', 'videos')
if os.path.exists(real_dir):
    real_videos = sorted([f for f in os.listdir(real_dir) if f.endswith('.mp4')])
    for vid_file in tqdm(real_videos, desc="İşleniyor: Original"):
        vid_path = os.path.join(real_dir, vid_file)
        stem = Path(vid_file).stem
        target_dir = os.path.join(DATA_ROOT, 'frames', 'original', stem)

        frame_paths = extract_frames(vid_path, target_dir, FRAMES_PER_VIDEO, IMG_SIZE)
        for fp in frame_paths:
            samples.append({'img': fp, 'label': 0, 'type': 'original'})

# JSON Kaydet
with open(os.path.join(LIST_DIR, 'dataset.json'), 'w') as f:
    json.dump(samples, f)

print(f"Tamamlandı! Toplam {len(samples)} kare ayıklandı.")
import os, json

DATASET_DIR = r'C:\Users\Tugba Gul\Desktop\roop_dataset'

samples = []

for img in os.listdir(os.path.join(DATASET_DIR, 'real')):
    if img.lower().endswith(('.jpg', '.jpeg', '.png')):
        samples.append({'img': os.path.join(DATASET_DIR, 'real', img), 'label': 0})

for img in os.listdir(os.path.join(DATASET_DIR, 'fake')):
    if img.lower().endswith(('.jpg', '.jpeg', '.png')):
        samples.append({'img': os.path.join(DATASET_DIR, 'fake', img), 'label': 1})

output_path = r'C:\Users\Tugba Gul\Desktop\faceforencis\roop_dataset.json'
with open(output_path, 'w') as f:
    json.dump(samples, f, indent=2)

print(f"Toplam: {len(samples)}")
print(f"Gerçek: {sum(1 for s in samples if s['label'] == 0)}")
print(f"Sahte:  {sum(1 for s in samples if s['label'] == 1)}")
print(f"Kaydedildi: {output_path}")
